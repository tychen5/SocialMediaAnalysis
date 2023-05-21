import os
import functools
import operator
import gc
import random
import time
import warnings

import cupy as cp
import numpy as np
import pandas as pd
import pickle
import timeout_decorator
import hdbscan as hdbscanpkg
import matplotlib.pyplot as plt
from collections import Counter
from cuml.manifold import UMAP
from cuml.preprocessing import normalize as cunorm
from cuml.cluster import HDBSCAN, DBSCAN
from kneed import DataGenerator, KneeLocator
from scipy.spatial.distance import cosine, euclidean
from scipy import stats
from scipy.stats import expon
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize as sknorm
from tqdm.auto import tqdm
from DBCV import DBCV

social_media_name = "enter_your_social_media_name"
used_params_li = []
param_path = "/path/to/your/notebooks/tmp_socialmedia_" + social_media_name + "_sentences_clustering_used_params_sentence_v0.10.pkl"

if os.path.exists(param_path):
    used_params_li = pickle.load(open(param_path, 'rb'))

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

labeled_df = pickle.load(open("/path/to/your/notebooks/social_media_" + social_media_name + "_sentence_final_v0.1_df.pkl", 'rb'))

try:
    labeled_df = labeled_df.drop(['sentiment_type', 'sentiment_type_xlm'], axis=1)
except KeyError:
    pass
def search_metric_weight_smalltopics(algo_li, weight1, weight2, weight3, weight4):
    """
    Use big topics' algo to evaluate in small topics
    """
    for topic in all_topics:
        if topic not in small_topic_li:
            topic_df = topic_df_di[topic]
            vectors = np.array(topic_df['Embeddings'].tolist())
            distance_matrix = pairwise_distances(vectors, vectors, metric='cosine', n_jobs=-1)
            algo_nameli = []
            algo_ansli = []
            algo_take = []
            algo_id_li = []
            algo_score_li = []
            for (algo_id, algo) in enumerate(algo_li):
                try:
                    if len(topic_df) <= max(algo.get_params()['min_samples'], algo.get_params()['min_cluster_size']):
                        algo_id_li.append(algo_id)
                        algo_score_li.append(np.nan)
                except KeyError:
                    if len(topic_df) <= algo.get_params()['min_samples']:
                        algo_id_li.append(algo_id)
                        algo_score_li.append(np.nan)
                metri = algo.metric
                params = algo.get_params()
                algo_nameli.append(params)
                if metri == 'precomputed':
                    X = cp.array(distance_matrix)
                else:
                    X = cunorm(cp.array(vectors))
                try:
                    labels = fit_algo(algo, X)
                except timeout_decorator.TimeoutError:
                    algo_take.append(algo)
                    algo_ansli.append(labels)
                    algo_id_li.append(algo_id)
                    del X
                    del labels
                    del algo
                    gc.collect()
            if len(algo_score_li) == len(algo_li):
                algo_df = pd.DataFrame(algo_nameli)
                algo_df['cluster_id'] = algo_ansli
                algo_df[['S_score', 'D_score', 'C_score', 'DBCVc_score', 'DBCVe_score', 'DBCV2c_score', 'DBCV2e_score',
                         'loner_score', 'toobig_score', 'group_num']] = algo_df.apply(
                    lambda x: calc_scores(vectors, x.cluster_id),
                    axis=1, result_type='expand')
                algo_df['D_norm'] = 2 * (
                        (algo_df['D_score'] - algo_df['D_score'].min()) / (
                        algo_df['D_score'].max() - algo_df['D_score'].min() + 1e-10)) - 1
                algo_df['C_norm'] = 2 * (
                        (algo_df['C_score'] - algo_df['C_score'].min()) / (
                        algo_df['C_score'].max() - algo_df['C_score'].min() + 1e-10)) - 1
                algo_df['loner_norm'] = 2 * (
                        (algo_df['loner_score'] - algo_df['loner_score'].min()) / (
                        algo_df['loner_score'].max() - algo_df['loner_score'].min() + 1e-10)) - 1
                algo_df['toobig_norm'] = 2 * (
                        (algo_df['toobig_score'] - algo_df['toobig_score'].min()) / (
                        algo_df['toobig_score'].max() - algo_df['toobig_score'].min() + 1e-10)) - 1
                algo_df['group_num'] = 2 * (
                        (algo_df['group_num'] - algo_df['group_num'].min()) / (
                        algo_df['group_num'].max() - algo_df['group_num'].min() + 1e-10)) - 1
                algo_df['final_score'] = (
                        algo_df['S_score'].astype(float) * weight1 - algo_df['D_norm'].astype(float) * weight1 + algo_df[
                    'C_norm'].astype(float) * weight1 + algo_df['DBCVc_score'].astype(float) * weight2 + algo_df[
                            'DBCVe_score'].astype(float) * weight2 + algo_df['DBCV2c_score'].astype(float) * (
                                weight2 + 0.5) + algo_df['DBCV2e_score'].astype(float) * (
                                weight2 + 0.5) - algo_df['loner_norm'].astype(float) * weight3 - algo_df[
                            'toobig_norm'].astype(float) * weight4 - algo_df['group_num'].astype(float) * (
                                (weight3 + weight4) / 10)) / (
                                           weight1 * 3 + weight2 * 2 + (weight2 + 0.5) * 2 + weight3 + weight4 + (
                                           weight3 + weight4) / 10)
                final_df = algo_df.copy()
                final_df['emb'] = [vectors] * len(final_df)
                final_df['weighted_score'] = final_df['final_score'] * small_topic_di[topic]
                topic_di[topic] = final_df
                score_li = final_df['weighted_score'].tolist()
                algo_score_li.extend(score_li)
                algo_id_li_, algo_score_li_ = zip(*sorted(zip(algo_id_li, algo_score_li)))
                take_score_li.append(list(algo_score_li_))
                del vectors
                del distance_matrix
                del topic_df
                gc.collect()
    score_df = pd.DataFrame(take_score_li, columns=[i for i in range(len(algo_li))])
    return topic_di, score_df


def eval_small_cluster(topic_di, smallcluster_scoredf):
    algo_idx = smallcluster_scoredf.mean()[smallcluster_scoredf.mean() == smallcluster_scoredf.mean().max()].index[0]
    for topic in all_topics:
        if topic not in small_topic_li:
            try:
                tmp = topic_di[topic]
            except KeyError:
                topic_di[topic] = tmp[tmp.index == algo_idx]
    return topic_di


def save_results(topic_di, output_dir_path, pickle_path):
    # Replace the following line with dummy paths
    # e.g., output_dir_path = '/path/to/your/output_dir'
    #       pickle_path = '/path/to/your/pickle_file'
    pass


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


def remapping_sentence_table(topic_sentence_df, labeled_df, same_index_di):
    """
    Note: Only provides visualization remap, the actual stored embedding vector/centroid vector/eigen vector will still be deduplicated
    labeled_df: Complete labeled_df (including duplicates)
    same_index_di: topic => key sent idx => same sent idx list as key
    """
    pass


def map_sentiment(ori_sent, sent_table, fullcomment_table):
    pass


def add_centroid_dist(topic_df_dict, cluster_df_dict, sentence_table_path, labeled_df):
    pass

import pandas as pd
import numpy as np
import random
from tqdm import tqdm

def refine_statistics(sentence_table, cluster_table_path, topic_table_path):
    """
    Recalculate statistics by grouping by sentence due to remapping.
    Add centroid sentence.
    """
    cluster_table = pd.read_csv(cluster_table_path)
    topic_table = pd.read_csv(topic_table_path)
    sentence_table_part = sentence_table[['topic_id', 'cluster_id', 'sent_translation', 'comment_sent', 'centroid_distance']]
    groupby_df = sentence_table_part.groupby(['topic_id', 'cluster_id', 'sent_translation']).agg(list).reset_index()
    count_df = pd.DataFrame(groupby_df['cluster_id'].value_counts()).reset_index()
    count_df.columns = ['cluster_id', 'sentence_num']
    count_df = count_df.sort_values(['cluster_id']).reset_index(drop=True)

    def refine_loner(topic_id, count_df):
        tid = str(topic_id)
        if len(tid) < 2:
            tid = '0' + tid
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

# Other functions (combine_with_duplicate, append_filtered_commsent) go here

# Main code
take_num = 10  # Replace with the desired number of iterations

for r in tqdm(range(take_num)):
    # Code for parameter selection and processing goes here

    # Save and process results
    topic_table, cluster_table = refine_statistics(sentence_table, cluster_table_path, topic_table_path)
    sentence_table = append_filtered_commsent(topic_statistic_df, df_full_comments, sentence_table, sentence_table_path)

    # Clear output for debugging purposes
    clear_output(wait=True)
