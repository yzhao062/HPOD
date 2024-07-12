# Meta-initialization

import os

import numpy as np
import pandas as pd
from pyod.utils.utility import standardizer, argmaxn
from scipy.stats import pearsonr

output_files = pd.read_csv(os.path.join('datasets', 'file_list.csv'), header=None).to_numpy().tolist()
output_files = [item for sublist in output_files for item in sublist]

n_datasets = len(output_files)

meta_features = np.load(os.path.join('datasets', 'meta_features.npy'))
meta_features = np.nan_to_num(meta_features)

# no need for standardization for tree models
meta_features = standardizer(meta_features)

random_seed = 42
random_state = np.random.RandomState(random_seed)

similarity_meta = np.zeros([n_datasets, n_datasets])

for i in range(n_datasets):
    for j in range(n_datasets):
        similarity_meta[i, j] = pearsonr(meta_features[i, :], meta_features[j, :])[0]

        if i == j:
            similarity_meta[i, j] = -1


def get_meta_init(n_init_models, ap_values, top1=False, return_weights=False):
    similar_datasets = []
    init_models = []

    if n_init_models >= 39:
        top1 = True

    # if use the top1 per similar dataset
    if not top1:
        for i in range(n_datasets):
            temp_models = []

            # first get the similar dataset
            similar_datasets.append(argmaxn(similarity_meta[i, :], n_init_models).tolist())

            for k in range(n_init_models):
                temp_models.append(np.argmax(ap_values[similar_datasets[-1][k], :]))

            init_models.append(temp_models)

    # if use the top k from the most similar dataset
    else:
        for i in range(n_datasets):
            # temp_models = []

            # first get the similar dataset
            similar_datasets.append(argmaxn(similarity_meta[i, :], 1).tolist())

            temp_models = argmaxn(ap_values[similar_datasets[-1][-1], :], n_init_models).tolist()

            # we need to trim the number of models
            init_models.append(random_state.choice(a=temp_models, size=n_init_models, replace=False).tolist())

    weights = []

    for i in range(n_datasets):
        # original
        weights.append(similarity_meta[i][similar_datasets[i]])

    if return_weights:
        return init_models, similar_datasets, weights
    else:
        return init_models, similar_datasets
