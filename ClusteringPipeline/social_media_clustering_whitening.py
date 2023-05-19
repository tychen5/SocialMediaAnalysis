
import os
import functools
import operator
import warnings
import random
import time
import multiprocessing as mp

import cupy as cp
import numpy as np
import pandas as pd
import pickle
import gc
import requests
import json

from cuml.preprocessing import normalize as cunorm
from sklearn.preprocessing import normalize as sknorm
from cuml.cluster import HDBSCAN, DBSCAN
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import pairwise_distances
from sklearn import metrics
from DBCV import DBCV
from scipy.spatial.distance import cosine, euclidean
import timeout_decorator
import hdbscan as hdbscanpkg
from scipy import stats
from scipy.stats import expon
import torch
from tqdm.auto import tqdm
from kneed import DataGenerator, KneeLocator
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, pairwise_kernels
from effcossim.pcs import pairwise_cosine_similarity, pp_pcs

from aitools.evaluation.ui_aim import UIAim

# Replace with your own paths and credentials
social_media_name = "Reddit"
param_path = "/path/to/your/params.pkl"
train_path = "/path/to/your/training_data.pkl"
csv_output_dir = "/path/to/your/csv_output/"
precompute_duplicate_path = "/path/to/your/precompute_duplicate.pkl"

# Load used_params_li if it exists
used_params_li = []
if os.path.exists(param_path):
    used_params_li = pickle.load(open(param_path, 'rb'))

warnings.filterwarnings('ignore')

# Initialize UIAim
uiaim = UIAim(
    experiment='220523_cluster_hdbscan_dbscan_socialmedia',
    task="reddit_train_whitening_new3_leo",
    training_data_dir=train_path
)

# Load labeled_df and remove unnecessary columns
labeled_df = pickle.load(open(train_path, 'rb'))
try:
    labeled_df = labeled_df.drop(['sentiment_type', 'sentiment_type_xlm'], axis=1)
except KeyError:
    pass

# Other preprocessing steps

def calc_scores(vectors, labels):
    """
    Calculate various clustering evaluation scores for the given vectors and labels.
    """
    idx_take = np.argwhere(labels != -1).squeeze()
    stat = stats.mode(idx_take)
    group_num = len(set(labels))

    if group_num <= 10:
        group_penalty = 0
    else:
        group_penalty = 1 - expon.pdf((group_num - 10) / 10, 0, 2)

    try:
        noise_len = len(np.argwhere(labels == -1).squeeze())
    except TypeError:
        noise_len = 0

    try:
        most_label, most_count = stat.mode[0], stat.count[0]
        count_penalty = most_count / (len(labels) - noise_len)
    except IndexError:
        most_count = 0
        count_penalty = 0

    noise_score = noise_len / len(labels)
    eval_labels = labels[idx_take]
    eval_vec = vectors[idx_take, :]

    try:
        S_score = metrics.silhouette_score(eval_vec, eval_labels, metric='cosine')
        D_score = metrics.davies_bouldin_score(eval_vec, eval_labels)
        C_score = metrics.calinski_harabasz_score(eval_vec, eval_labels)
    except ValueError:
        S_score = 0
        D_score = 0
        C_score = 0

    try:
        hdbscan_score_c, hdbscan_score_e = hdbscan_scorer(eval_vec, eval_labels)
    except timeout_decorator.TimeoutError:
        hdbscan_score_c = 0
        hdbscan_score_e = 0

    try:
        hdbscan_score2_c, hdbscan_score2_e = hdbscan_scorer2(vectors, labels)
    except timeout_decorator.TimeoutError:
        hdbscan_score2_c = 0
        hdbscan_score2_e = 0

    return (S_score, D_score, C_score, hdbscan_score_c, hdbscan_score_e,
            hdbscan_score2_c, hdbscan_score2_e, noise_score, count_penalty, group_penalty)



def convert_str(li):
    final_li = []

    for i in li:
        i = str(i)
        if len(i) == 1:
            final_li.append(i + '.00')
        elif len(i) == 3:
            final_li.append(i + '0')
        else:
            final_li.append(i)

    return "_".join(final_li)


def refine_statistics(sentence_table, cluster_table_path, topic_table_path):
    """
    Recalculate statistics by grouping by sentence due to remapping.
    Add centroid sentence to the statistics.

    Args:
        sentence_table (DataFrame): The sentence table.
        cluster_table_path (str): The path to the cluster table.
        topic_table_path (str): The path to the topic table.
    Returns:
        tuple: A tuple containing the updated topic and cluster tables.
    """
    cluster_table = pd.read_csv(cluster_table_path)
    topic_table = pd.read_csv(topic_table_path)
    sentence_table_part = sentence_table[['topic_id', 'cluster_id', 'sent_translation', 'comment_sent', 'centroid_distance']]
    groupby_df = sentence_table_part.groupby(['topic_id', 'cluster_id', 'sent_translation']).agg(list).reset_index()
    count_df = pd.DataFrame(groupby_df['cluster_id'].value_counts()).reset_index()
    count_df.columns = ['cluster_id', 'sentence_num']
    count_df = count_df.sort_values(['cluster_id']).reset_index(drop=True)

    def refine_loner(topic_id, count_df):
        tid = str(topic_id).zfill(2)
        count_topic_df = count_df[count_df['cluster_id'].str.startswith(tid)]
        topic_sent_num = count_topic_df['sentence_num'].sum()
        topic_loner_num = count_topic_df[count_topic_df['cluster_id'].str.endswith('-1')]['sentence_num'].iloc[0]
        loner_ratio = topic_loner_num / topic_sent_num
        return str(topic_loner_num), loner_ratio

    topic_table[['loner_size', 'loner_ratio']] = topic_table.apply(lambda x: refine_loner(x.topic_id, count_df), axis=1, result_type='expand')
    topic_table.to_csv(topic_table_path, index=False)

    def refine_cluster_statistics(cluster_id, count_df):
        tid = cluster_id.split('_')[0]
        count_topic_df = count_df[count_df['cluster_id'].str.startswith(tid)]
        topic_sent_num = count_topic_df['sentence_num'].sum()
        topic_cluster_num = count_topic_df[count_topic_df['cluster_id'] == cluster_id]['sentence_num'].iloc[0]
        cluster_ratio = topic_cluster_num / topic_sent_num
        return str(topic_cluster_num), cluster_ratio

    cluster_table[['cluster_size', 'cluster_ratio']] = cluster_table.apply(lambda x: refine_cluster_statistics(x.cluster_id, count_df), result_type='expand', axis=1)
    groupby_df['centroid_distance'] = groupby_df['centroid_distance'].apply(lambda x: x[0])

    def find_nearest_centroid_sent(cid, groupby_df):
        cluster_sent_df = groupby_df[groupby_df['cluster_id'] == cid]
        min_dist = cluster_sent_df['centroid_distance'].min()
        return cluster_sent_df[cluster_sent_df['centroid_distance'] == min_dist]['sent_translation'].iloc[0]

    cluster_table['centroid_sentence'] = cluster_table['cluster_id'].apply(find_nearest_centroid_sent, args=(groupby_df,))
    cluster_table.to_csv(cluster_table_path, index=False)
    return topic_table, cluster_table

def combine_with_duplicate(vectors, labels, tid_cid_li, topic_oridf, topic_dup_di, no_duplicate_idx):
    """
    Combine vectors, labels, and tid_cid_li with their duplicates.

    Args:
        vectors (2D array): N x 768 dimensional array.
        labels (1D array): N dimensional array.
        tid_cid_li (list): List of N strings.
        topic_oridf (DataFrame): Topic's label_df with duplicates.
        topic_dup_di (dict): Topic's duplicate location index, key idx -> duplicate idx list.
        no_duplicate_idx (list): List of indices without duplicates.
    Returns:
        tuple: A tuple containing the combined vectors, labels, and tid_cid_li.
    """
    add_vecs = []
    add_labs = []
    add_tidcids = []
    tid_cid_arr = np.array(tid_cid_li)
    clean_idx_di = {}  # Reconstruct

    for key, value in topic_dup_di.items():
        value_li = value.copy()
        need_reconstruct = False
        for k, v in clean_idx_di.items():
            if key in v:
                need_reconstruct = True
                v_li = v.copy()
                v_li.extend(value_li)
                clean_idx_di[k] = list(set(v_li))
        if need_reconstruct == False:
            clean_idx_di[key] = value

    for k_idx, idx_li in clean_idx_di.items():
        try:
            key_idx = no_duplicate_idx.index(k_idx)
        except ValueError:
            continue
        key_label = labels[key_idx]
        key_tidcid = tid_cid_arr[key_idx]
        val_length = len(idx_li)
        val_labels = [key_label] * val_length
        val_tidcid = [key_tidcid] * val_length
        val_vecs = topic_oridf.loc[idx_li]['Emb_reduct'].tolist()
        add_vecs.extend(val_vecs)
        add_labs.extend(val_labels)
        add_tidcids.extend(val_tidcid)

    try:
        all_vectors = np.concatenate((vectors, add_vecs))
        all_labels = np.concatenate((labels, add_labs))
        all_tidcids = list(np.concatenate((tid_cid_arr, add_tidcids)))
    except ValueError:  # No duplicates
        all_vectors = vectors
        all_labels = labels
        all_tidcids = tid_cid_arr

    return all_vectors, all_labels, all_tidcids

import random
import gc
import numpy as np
import pandas as pd
import pickle
from IPython.display import clear_output

def append_filtered_commsent(topic_statistic_df, df_full_comments, sentence_table, sentence_table_path):
    """Append filtered comments and sentences to the sentence table.

    Args:
        topic_statistic_df (DataFrame): Topic statistics DataFrame.
        df_full_comments (DataFrame): Full comments DataFrame.
        sentence_table (DataFrame): Sentence table DataFrame.
        sentence_table_path (str): Path to the sentence table file.

    Returns:
        DataFrame: Updated sentence table.
    """
    topicmap_di = {}
    topic_name_li = topic_statistic_df['topic_name'].tolist()
    topic_id_li = topic_statistic_df['topic_id'].tolist()
    for name, idx in zip(topic_name_li, topic_id_li):
        topicmap_di[name] = idx
    diff_comm_idli = sorted(set(df_full_comments['comment_id'].unique()) - set(sentence_table['comment_id'].unique()))
    for cid in diff_comm_idli:
        missing_df = df_full_comments[df_full_comments['comment_id'] == cid]
        missing_df = missing_df[['created', 'published_at', 'language', 'comment_id',
                                 'upvote_ratio', 'ori_comment', 'translation', 'sentiment_overalltype', 'sentiment_overallscore',
                                 ]]
        missing_df.columns = ['created', 'published_at', 'oricomment_lang', 'comment_id',
                              'upvote_ratio', 'comment_sent', 'sent_translation', 'sentiment_overalltype', 'sentiment_overallscore']
        try:
            topic_id = topicmap_di['UNK']
            tid = str(topic_id)
            if len(tid) < 2:
                tid = '0' + tid
            cid = tid + '_-1'
            missing_df['topic_id'] = topicmap_di['UNK']
            missing_df['cluster_id'] = cid
        except KeyError:
            missing_df['topic_id'] = np.nan
            missing_df['cluster_id'] = np.nan
        missing_df['varianceavg_distance'] = np.nan
        missing_df['centroid_distance'] = np.nan
        sentence_table = sentence_table.append(missing_df, ignore_index=True)
    sentence_table.to_csv(sentence_table_path, index=False)
    return sentence_table

# Dummy values for sensitive information
param_tradmetric_li = ['enter_your_tradmetric']
param_dbmetric_li = ['enter_your_dbmetric']
param_penaltyloner_li = ['enter_your_penaltyloner']
param_penaltybiggest_li = ['enter_your_penaltybiggest']
param_path = '/path/to/your/params'

random.shuffle(param_tradmetric_li)
random.shuffle(param_dbmetric_li)
random.shuffle(param_penaltyloner_li)
random.shuffle(param_penaltybiggest_li)
take_num = (len(param_tradmetric_li) * len(param_dbmetric_li) * len(param_penaltyloner_li) * len(param_penaltybiggest_li)) * 2

# Dummy function for uiaim.SetTotalEpoch
def set_total_epoch(take_num):
    pass

set_total_epoch(take_num)
