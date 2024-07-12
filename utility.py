# Utility functions
import numpy as np
from scipy.stats import rankdata, kendalltau
from scipy.stats import wilcoxon


def get_sim_kendall(ind, ind_scores, model_list, n_sim, n_datasets, ap_values):
    kendall_sim = []
    for i in range(n_datasets):
        if i != ind:
            kendall_sim.append(kendalltau(ap_values[ind, model_list], ap_values[i, model_list])[0])
        else:
            kendall_sim.append(-1)
    kendall_sim_sort = np.argsort(np.asarray(kendall_sim) * -1)

    return kendall_sim_sort[:n_sim].tolist(), np.asarray(kendall_sim)[kendall_sim_sort[:n_sim].tolist()]


def get_rank(ap_score, our_result):
    all_ranks = rankdata(ap_score * -1, axis=1)

    rank_list = []

    for j in range(len(our_result)):
        rank_list.append(all_ranks[j, np.argmin(np.abs(ap_score[j, :] - our_result[j]))])

    return rank_list


def argmaxatn(w, nth):
    w = np.asarray(w).ravel()
    t = np.argsort(w)
    return t[-1 * nth]


def get_diff(li1, li2):
    return (list(set(li1) - set(li2)))


def get_permutations(list1, list2):
    full_perm = []
    for i in list1:
        for j in list2:
            full_perm.append((i, j))

    return full_perm


def get_dataset_similarity(ap1, ap2):
    # should write as pairwise, but use scipy for now
    return kendalltau(ap1, ap2)[0]


def get_dataset_similarity_pair(pair_list):
    return np.sum(pair_list) / len(pair_list)


def get_normalized_ap_diff(ap_diff):
    max_ap_diff = np.max(np.abs(ap_diff))
    return ap_diff / (max_ap_diff + 0.00000001)


def weighted_kendall_from_pairs(a, b):
    c1_ind = np.abs(a) <= np.abs(b)
    c2_ind = np.abs(a) > np.abs(b)
    c1 = a / (b + 0.0000001)
    c2 = b / (a + 0.0000001)
    c = np.zeros([len(a), ])
    c[c1_ind] = c1[c1_ind]
    c[c2_ind] = c2[c2_ind]

    return np.sum(c) / np.sum(np.abs(c))


def flatten(a):
    return [item for sublist in a for item in sublist]


def process_batch(n_hp_configs, sorted_ap_val, trials, n_trials):
    qth = []
    for k in range(n_trials):
        qth.append(get_qth(n_hp_configs, sorted_ap_val, trials[k]))

    return int(np.mean(qth)), np.round(np.mean(trials), decimals=2)


def get_qth(n_hp_configs, sorted_ap_val, our_result, verbose=False):
    our_qth = 0
    for i in range(n_hp_configs):
        if verbose:
            print('top', i + 1,
                  wilcoxon(sorted_ap_val[:, n_hp_configs - i - 1], our_result, alternative='greater'),
                  np.mean(sorted_ap_val[:, n_hp_configs - i - 1]),
                  np.mean(our_result))
        if wilcoxon(sorted_ap_val[:, n_hp_configs - i - 1], our_result, alternative='greater')[1] >= 0.05:
            our_qth = i + 1
            return our_qth
