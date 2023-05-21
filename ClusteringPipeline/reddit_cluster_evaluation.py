#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import chisquare
from scipy.special import kl_div


def calc_statistics(li):
    return np.mean(li), np.std(li)


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


def my_dist(mean, std, cluster_size_li):
    std2_upbound = mean + 2 * std
    std3_upbound = mean + 3 * std
    std2_lowbound = mean - 2 * std
    std3_lowbound = mean - 3 * std
    li = np.array(cluster_size_li)
    ori_li_len = len(li)
    idx = np.where(li < std3_upbound)[0]
    idx_num = len(idx)
    if idx_num == len(li):
        idx = np.where(li < std2_upbound)[0]
        idx_num = len(idx)
        if idx_num == len(li):
            idx = np.where(li < np.max(li))[0]
            idx_num = len(idx)
    li = li[idx]
    idx = np.where(li > std3_lowbound)[0]
    idx_num = len(idx)
    if idx_num == len(li):
        idx = np.where(li > std2_lowbound)[0]
        idx_num = len(idx)
        if idx_num == len(li):
            idx = np.where(li > np.min(li))[0]
            idx_num = len(idx)
    li = li[idx]
    sent_num = np.sum(li)
    value = sent_num / ori_li_len
    return value


def clean(path):
    return path.split('appreview_network_clustering_tuples')[-1].split('.pkl')[0]


# Replace the following paths with your own paths
dir_path = '/path/to/your/data/'
csv_path = '/path/to/your/csv/'

pkl_path_li = next(os.walk(dir_path))[2]
cluster_num = []
cluster_allsize_li = []
chi_square = []
chi_square2 = []
kl_div_li = []
original_cluster_sizes = []
for pkl_path in pkl_path_li:
    one, two, three, four = pickle.load(open(dir_path + pkl_path, 'rb'))
    all_length = 0
    cluster_length = 0
    cluster_lengthli = []
    chi_li = []
    cluster_size_li = []
    topic_size_li = []
    for key in two.keys():
        tmpdf = two[key]
        all_length = all_length + len(tmpdf)
        cluster_lengthli.extend(tmpdf['cluster_size'].tolist())
        cluster_size = tmpdf['cluster_size'].sum()
        cluster_size_li.append(cluster_size)
        topic_size = three[three['topic_name'] == key]['sentences_num'].iloc[0]
        topic_size_li.append(topic_size)
        cluster_length = cluster_length + cluster_size
    sum_clustersize = np.sum(cluster_size_li)
    sum_topicsize = np.sum(topic_size_li)
    kl_div_value = kl_div(topic_size_li, cluster_size_li).sum()
    cluster_size_li = [x / sum_clustersize for x in cluster_size_li]
    topic_size_li = [x / sum_topicsize for x in topic_size_li]
    p_value = chisquare(cluster_size_li, f_exp=topic_size_li)
    cluster_num.append(all_length)
    cluster_allsize_li.append(cluster_length)
    chi_square.append(p_value[0])
    chi_square2.append(str(p_value[1]))
    original_cluster_sizes.append(cluster_lengthli)
    kl_div_li.append(kl_div_value)

pkl_path_li_ = [x.split('appreview_network_clustering_tuples')[-1].split('.pkl')[0] for x in pkl_path_li]
stat_df = pd.DataFrame(pkl_path_li, columns=['pkl_path'])
stat_df['cluster_num_total'] = cluster_num
stat_df['cluster_sentences_num_total'] = cluster_allsize_li
stat_df['chi_square'] = chi_square
stat_df['chi_pvalue'] = chi_square2
stat_df['kl_div'] = kl_div_li
stat_df['cluster_all_sizes'] = original_cluster_sizes

stat_df[['clustersize_mean', 'cluster_size_std']] = stat_df.apply(lambda x: calc_statistics(x.cluster_all_sizes), axis=1, result_type='expand')
stat_df[['2std_value', '3std_value']] = stat_df.apply(lambda x: calc_dist(x.clustersize_mean, x.cluster_size_std, x.cluster_all_sizes), axis=1, result_type='expand')
stat_df.to_csv(csv_path + 'evaluation_statistiscs.csv', index=False)
stat_df['Leo_index'] = stat_df.apply(lambda x: my_dist(x.clustersize_mean, x.cluster_size_std, x.cluster_all_sizes), axis=1)

tmp_df = stat_df[(stat_df['Leo_index'] >= 81.40625 * 0.924) & (stat_df['Leo_index'] < 98.25 * 1.076)]
tmp_df['parameters'] = tmp_df['pkl_path'].apply(clean)
tmp_df.to_csv(csv_path + 'Leo_select_parameters.csv', index=False)

get = tmp_df['pkl_path'].tolist()
h_min_sample_li = []
h_min_cluster_li = []
d_min_sample_li = []
d_eps_li = []
for pkl_path in get:
    one, two, three, four = pickle.load(open(dir_path + pkl_path, 'rb'))
    for key in four.keys():
        tmpdf = four[key]
        if tmpdf.iloc[0]['metric'] == 'precomputed':
            d_min_sample_li.extend(tmpdf['min_samples'].tolist())
            d_eps_li.extend(tmpdf['eps'].tolist())
        else:
            h_min_sample_li.extend(tmpdf['min_samples'].tolist())
            h_min_cluster_li.extend(tmpdf['min_cluster_size'].tolist())

print('DBSCAN:')
print('min_samples:', min(d_min_sample_li), max(d_min_sample_li), sorted(set(d_min_sample_li)))
print('eps:', min(d_eps_li), max(d_eps_li), sorted(set(d_eps_li)))
print('HDBSCAN:')
try:
    print('min_samples:', min(h_min_sample_li), max(h_min_sample_li), sorted(h_min_sample_li))
    print('min_cluster_size:', min(h_min_cluster_li), max(h_min_cluster_li), sorted(h_min_cluster_li))
except ValueError:
    print('No HDBSCAN!!')
