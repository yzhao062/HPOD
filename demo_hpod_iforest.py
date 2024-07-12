# Demo of using HPOD on iForest
import os
import time
from copy import deepcopy

import lightgbm as lgb
import numpy as np
import pandas as pd
from combo.models.score_comb import average
from pyod.utils.utility import standardizer
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import KFold

from init_meta import get_meta_init
from utility import get_diff, process_batch, get_sim_kendall

if __name__ == "__main__":

    # load the 39 file list
    file_list = pd.read_csv(os.path.join('datasets','file_list.csv'), header=None).to_numpy().tolist()
    file_list = [item for sublist in file_list for item in sublist]
    n_datasets = len(file_list)

    # size of the meta-HP set
    n_hp_configs = 288

    # load the precomputed features for fast demo
    X = np.load(os.path.join('datasets', 'precomputed', 'iforest', 'X.npy'))
    X = X.astype(np.float32)
    X = np.nan_to_num(X)

    X_inner = np.load(os.path.join('datasets', 'precomputed', 'iforest', 'X_inner.npy'))
    X_inner = X_inner.astype(np.float32)
    X_inner = np.nan_to_num(X_inner)

    # add standardization
    X, X_inner = standardizer(X, X_inner)
    X = np.nan_to_num(X)
    X_inner = np.nan_to_num(X_inner)

    # load ap rank and ap values
    y = np.load(os.path.join('datasets', 'precomputed', 'iforest', 'y_ap_rank.npy'))  # for ap rank
    ap_values = y.reshape(n_datasets, n_hp_configs)
    ap_values_inner = pd.read_csv(os.path.join('datasets', 'precomputed', 'iforest', 'ap_rank_inner.csv'),
                                  header=None).to_numpy()
    sorted_ap_val = np.sort(ap_values, axis=1)
    ap_all = np.concatenate([ap_values, ap_values_inner], axis=1)

    # set up the random seed
    random_seed = 42
    random_state = np.random.RandomState(random_seed)

    # set the HPs of HPOD
    n_trials = 1
    n_epochs = 30

    n_init_configs = 10
    n_neighbors = 1

    # specify the input features, i.e., HPs, for the surrogate function
    feature_index = list(range(200, 203))

    # LOOCV; build the train test index. One dataset for test
    kf = KFold(n_splits=(int(n_datasets)))

    train_indexs = []
    test_indexs = []

    inds = []
    all_index = np.arange(X.shape[0])

    counter = 0
    for train_index, test_index in kf.split(X):
        inds.append(counter)
        train_indexs.append(train_index)
        test_indexs.append(test_index)
        counter += 1

    # Build meta-surrogate functions
    gp_clf = GaussianProcessRegressor()
    pre_clfs = []
    for ind, train_index, test_index in zip(inds, train_indexs, test_indexs):
        print('Build meta-surrogate function', ind)

        X_train, X_test = X[train_index, :], X[test_index, :]
        y_train, y_test = y[train_index], y[test_index]

        pre_clfs.append(deepcopy(gp_clf).fit(X_test[:, feature_index], y_test))

    # track HPOD's performance
    ours = []

    best_model_history = np.zeros([n_datasets, n_epochs]).astype('float')
    best_model_index = np.zeros([n_datasets, n_epochs]).astype('int')
    ei_tracker = np.zeros([n_datasets, n_epochs]).astype('float')

    init_configs_list, _ = get_meta_init(n_init_models=n_init_configs, ap_values=ap_values)
    meta_list, _ = get_meta_init(n_init_models=n_epochs, ap_values=ap_values)

    for k in range(n_trials):

        for ind, train_index, test_index in zip(inds, train_indexs, test_indexs):

            meta_train_list = get_diff(list(range(n_datasets)), [ind])

            # meta init
            init_configs = init_configs_list[ind]
            # remove duplicates
            init_configs = list(set(init_configs))

            # initialize the evaluation set
            all_models = list(range(ap_values.shape[1] + ap_values_inner.shape[1]))
            curr_models = init_configs
            left_models = get_diff(all_models, curr_models)

            X_train, X_test = X[train_index, :], X[test_index, :]
            y_train, y_test = y[train_index], y[test_index]

            # !!!!!!#
            X_test_inner = X_inner[test_index, :]
            X_test_all = np.concatenate([X_test, X_test_inner], axis=0)
            # X_test_all = standardizer(X_test_all)

            # warm up the surrogate
            X_test_s = X_test_all[curr_models,]
            y_test_s = y_test[curr_models]

            # load the performance evaluator f(.)
            clf = lgb.Booster(model_file=os.path.join('models', 'iforest', 'rgb_' + str(ind) + '.pt'))

            f_pred = clf.predict(X_test_s)

            ######################
            # # train s on it -> selected train and the corresponding f results
            start = time.time()
            rfu = GaussianProcessRegressor()

            # only use HP configs.
            rfu.fit(X_test_s[:, feature_index], f_pred)

            for epoch in range(n_epochs):

                # this should be changed to f's prediction
                curr_model_best_idx = np.argmax(f_pred)
                curr_model_best = np.max(f_pred)

                # update neighbor and weights for surrogate transfer
                neighbors, weights = get_sim_kendall(ind, f_pred, curr_models, n_neighbors, n_datasets, ap_all)

                mu_list, sigma_list = rfu.predict(X_test_all[left_models, :][:, feature_index], return_std=True)

                neighbor_mu = []
                neighbor_sigma = []
                for n in neighbors:
                    # !!!!!!!#
                    mu_temp, sigma_temp = pre_clfs[n].predict(X_test_all[left_models, :][:, feature_index],
                                                              return_std=True)
                    neighbor_mu.append(mu_temp)
                    neighbor_sigma.append(sigma_temp)

                neighbor_mu_avg = average(np.asarray(neighbor_mu).T, weights.reshape(1, -1))

                mu_list = mu_list + neighbor_mu_avg

                z_list = (mu_list - curr_model_best) / sigma_list
                ei = (mu_list - curr_model_best) * norm.cdf(z_list) + sigma_list * norm.pdf(z_list)
                ei[np.where(sigma_list == 0)] = 0

                assert (len(ei) == len(left_models))

                # next best model
                ei_max = np.argmax(ei)
                next_model = left_models[ei_max]

                # refit the surrogate
                curr_models.append(next_model)
                left_models.remove(next_model)
                assert (len(curr_models) + len(left_models) == len(all_models))

                X_test_s = X_test_all[curr_models, :]
                y_test_s = clf.predict(X_test_s)

                f_pred = y_test_s

                # retrain s on it -> selected train and the corresponding f results
                rfu.fit(X_test_s[:, feature_index], f_pred)

                best_model_history[ind, epoch] = ap_all[ind, curr_models[curr_model_best_idx]]
                best_model_index[ind, epoch] = curr_models[curr_model_best_idx]

                ei_tracker[ind, epoch] = ei.max()

            ours.append(ap_all[ind, curr_models[curr_model_best_idx]])
            print('HPOD trail', k, ind, 'identified HP avg. norm. rank', ours[-1])

    ours_all = np.asarray(ours).reshape(n_trials, n_datasets)
    ours = np.mean(ours_all, axis=0).tolist()

    ours_qth, ours_avg = process_batch(n_hp_configs, sorted_ap_val, ours_all, n_trials)

    print('*****************************************************************')
    print('Performance summary:')
    print('HPOD top-qth across 39 dataset', ours_qth)
    print('HPOD avg. normalized rank across 39 dataset', ours_avg)
