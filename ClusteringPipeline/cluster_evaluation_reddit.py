
import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import chisquare
from scipy.special import kl_div

social_media_name = "Reddit"
dir_path = './Data/your_data_path/'  # Update with your data path
csv_path = './Results/your_results_path/'  # Update with your results path
pkl_path_li = next(os.walk(dir_path))[2]

# Initialize lists for storing results
cluster_num = []
cluster_allsize_li = []
chi_square = []
chi_square2 = []
kl_div_li = []
original_cluster_sizes = []

# Iterate through pickle files
for pkl_path in pkl_path_li:
    one, two, three, four = pickle.load(open(dir_path + pkl_path, 'rb'))
    all_length = 0
    cluster_length = 0
    cluster_lengthli = []
    chi_li = []
    cluster_size_li = []
    topic_size_li = []

    # Iterate through topics
    for key in two.keys():
        tmpdf = two[key]
        all_length += len(tmpdf)
        cluster_lengthli.extend(tmpdf['cluster_size'].tolist())
        cluster_size = tmpdf['cluster_size'].sum()
        cluster_size_li.append(cluster_size)
        topic_size = three[three['topic_name'] == key]['sentences_num'].iloc[0]
        topic_size_li.append(topic_size)
        cluster_length += cluster_size

    sum_clustersize = np.sum(cluster_size_li)
    sum_topicsize = np.sum(topic_size_li)
    kl_div_value = kl_div(topic_size_li, cluster_size_li).sum()
    cluster_size_li = [x / sum_clustersize for x in cluster_size_li]
    topic_size_li = [x / sum_topicsize for x in topic_size_li]
    p_value = chisquare(cluster_size_li, f_exp=topic_size_li)

    # Append results to lists
    cluster_num.append(all_length)
    cluster_allsize_li.append(cluster_length)
    chi_square.append(p_value[0])
    chi_square2.append(str(p_value[1]))
    original_cluster_sizes.append(cluster_lengthli)
    kl_div_li.append(kl_div_value)

pkl_path_li_ = [x.split('social_media_clustering_tuples_')[-1].split('.pkl')[0] for x in pkl_path_li]

# Create DataFrame for statistics
stat_df = pd.DataFrame(pkl_path_li, columns=['pkl_path'])
stat_df['cluster_num_total'] = cluster_num
stat_df['cluster_sentences_num_total'] = cluster_allsize_li
stat_df['chi_square'] = chi_square
stat_df['chi_pvalue'] = chi_square2
stat_df['kl_div'] = kl_div_li
stat_df['cluster_all_sizes'] = original_cluster_sizes

# Calculate statistics
def calc_statistics(li):
    return np.mean(li), np.std(li)

stat_df[['clustersize_mean', 'cluster_size_std']] = stat_df.apply(lambda x: calc_statistics(x.cluster_all_sizes), axis=1, result_type='expand')

# Calculate distance
def calc_dist(mean, std, cluster_size_li):
    std2_bound = mean + 2 * std
    std3_bound = mean + 3 * std
    li = np.array(cluster_size_li)
    li2_idx = np.where(li < std2_bound)[0]
    li2_cluster_num = len(li2_idx)
    li2_sentence_sum = np.sum((li[li2_idx]))
    li2_value = li2_sentence_sum / li2_cluster_num
    li3_idx = np.where(li < std3_bound)[0]
    li3_cluster_num = len(li3_idx)
    li3_sentence_sum = np.sum((li[li3_idx]))
    li3_value = li3_sentence_sum / li3_cluster_num
    return li2_value, li3_value

stat_df[['2std_value', '3std_value']] = stat_df.apply(lambda x: calc_dist(x.clustersize_mean, x.cluster_size_std, x.cluster_all_sizes), axis=1, result_type='expand')

# Save statistics to CSV
stat_df.to_csv(csv_path + 'evaluation_statistiscs.csv', index=False)

import numpy as np
import torch
from scipy import spatial
from numba import jit
import numba
import tensorflow as tf
import faiss
import cuml
from cuml.metrics import pairwise_distances


def sim_matrix(a, b, eps=1e-6):
    """
    Compute the similarity matrix between two tensors a and b.

    Args:
        a (torch.Tensor): First tensor.
        b (torch.Tensor): Second tensor.
        eps (float, optional): Small value for numerical stability. Defaults to 1e-6.

    Returns:
        torch.Tensor: Similarity matrix.
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return 1 - sim_mt


def compute_cosine_distances(a, b):
    """
    Compute the cosine distances between two tensors a and b using TensorFlow.

    Args:
        a (tf.Tensor): First tensor.
        b (tf.Tensor): Second tensor.

    Returns:
        tf.Tensor: Cosine distances.
    """
    normalize_a = tf.nn.l2_normalize(a, 1)
    normalize_b = tf.nn.l2_normalize(b, 1)
    distance = 1 - tf.matmul(normalize_a, normalize_b, transpose_b=True)
    return distance


def main():
    # Example usage of sim_matrix
    input1 = torch.randn(5, 5)
    sim_mt = sim_matrix(input1, input1)
    print(sim_mt)

    # Example usage of compute_cosine_distances
    X = np.random.uniform(0, 10, (100, 512)).astype('float32')
    X = tf.constant(X)
    cosine_distances = compute_cosine_distances(X, X)
    print(cosine_distances)


