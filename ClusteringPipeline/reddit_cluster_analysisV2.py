import os
import functools
import operator
import warnings
import random
import time
import gc
import json
import requests
import numpy as np
import pandas as pd
import pickle
import cupy as cp
import matplotlib.pyplot as plt
from collections import Counter
from cuml.manifold import UMAP
from cuml.preprocessing import normalize as cunorm
from sklearn.preprocessing import normalize as sknorm
from cuml.cluster import HDBSCAN, DBSCAN
from sklearn.metrics import pairwise_distances
from sklearn import metrics
from DBCV import DBCV
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
from timeout_decorator import timeout
import hdbscan as hdbscanpkg
from scipy import stats
from scipy.stats import expon
from tqdm.auto import tqdm
from kneed import DataGenerator, KneeLocator
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_kernels
from effcossim.pcs import pairwise_cosine_similarity, pp_pcs

social_media_name = "Reddit"
used_params_li = []  # Used weight ratios
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

# ... rest of the code ...

import os
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import pairwise_distances


def save_results(topic_di, output_dir_path, pickle_path):
    comment_topic_di = {}
    topic_cluster_di = {}
    cluster_stat_li = []
    calc_overall_score = []

    for t_id, topic in enumerate(topic_di.keys()):
        # Process each topic
        # ...
        "PLEASE INSERT YOUR OWN TEXT"
    # Save results
    output_dir_path = output_dir_path + '_' + cluster_num + '_' + cluster_size_total + '_' + final_score
    if not os.path.isdir(output_dir_path):
        os.makedirs(output_dir_path)

    # Save tables as CSV files
    # ...

    # Save pickle file
    pkl_dir = "/".join(pickle_path.split("/")[:-1])
    if not os.path.isdir(pkl_dir):
        os.makedirs(pkl_dir)
    pickle.dump(obj=(comment_topic_di, topic_cluster_di, topic_statistic_df, topic_di),
                file=open(pickle_path + '_' + final_score + '.pkl', 'wb'))

    return comment_table, cluster_table, topic_statistic_df, comment_table_path, cluster_table_path, topic_table_path, pickle_path + '_' + final_score + '.pkl'


def convert_str(li):
    # ...
    return "_".join(final_li)


def remapping_sentence_table(topic_sentence_df, labeled_df, same_index_di):
    """
    Remap sentence table for visualization purposes.
    """
    # ...


def map_sentiment(ori_sent, sent_table, fullcomment_table):
    # ...
    return sent_sent_type, sent_sent_score, fullcomment, comm_sent_type, comm_sent_score


def add_centroid_dist(topic_df_dict, cluster_df_dict, sentence_table_path, labeled_df):
    # ...
    "PLEASE INSERT YOUR OWN CODE"


def refine_statistics(sentence_table, cluster_table_path, topic_table_path):
    """
    Refine statistics due to remapping.
    """
    # ...


def combine_with_duplicate(vectors, labels, tid_cid_li, topic_oridf, topic_dup_di, no_duplicate_idx):
    """
    Combine vectors, labels, and tid_cid_li with duplicates.
    """
    # ...

import random
import numpy as np
import pandas as pd
import pickle
from IPython.display import clear_output

def append_filtered_commsent(topic_statistic_df, df_full_comments, sentence_table, sentence_table_path):
    """
    Append filtered comments and sentences to the sentence table.

    Args:
        topic_statistic_df (DataFrame): DataFrame containing topic statistics.
        df_full_comments (DataFrame): DataFrame containing full comments.
        sentence_table (DataFrame): DataFrame containing sentence information.
        sentence_table_path (str): Path to save the updated sentence table.
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
        missing_df = missing_df[['App Store', 'App Name', 'version', 'published_at', 'language', 'comment_id',
                                 'rating', 'ori_comment', 'translation', 'sentiment_overalltype', 'sentiment_overallscore']]
        missing_df.columns = ['App Store', 'App Name', 'version', 'published_at', 'oricomment_lang', 'comment_id',
                              'rating', 'comment_sent', 'sent_translation', 'sentiment_overalltype', 'sentiment_overallscore']
        topic_id = topicmap_di['UNK']
        tid = str(topic_id)
        if len(tid) < 2:
            tid = '0' + tid
        cid = tid + '_-1'
        missing_df['topic_id'] = topicmap_di['UNK']
        missing_df['cluster_id'] = cid
        missing_df['varianceavg_distance'] = np.nan
        missing_df['centroid_distance'] = np.nan
        sentence_table = sentence_table.append(missing_df, ignore_index=True)
    
    sentence_table.to_csv(sentence_table_path, index=False)
    return sentence_table

# Dummy values for parameters
param_tradmetric_li = [1, 2, 3]
param_dbmetric_li = [1, 2, 3]
param_penaltyloner_li = [1, 2, 3]
param_penaltybiggest_li = [1, 2, 3]

random.shuffle(param_tradmetric_li)
random.shuffle(param_dbmetric_li)
random.shuffle(param_penaltyloner_li)
random.shuffle(param_penaltybiggest_li)

take_num = (len(param_tradmetric_li) * len(param_dbmetric_li) * len(param_penaltyloner_li) * len(param_penaltybiggest_li)) * 2

for r in range(take_num):
    try:
        used_params_li = pickle.load(open('path/to/your/params.pkl', 'rb'))
    except FileNotFoundError:
        used_params_li = []

    param1 = random.choice(param_tradmetric_li)
    param2 = random.choice(param_dbmetric_li)
    param3 = random.choice(param_penaltyloner_li)
    param4 = random.choice(param_penaltybiggest_li)
    params = [round(param1 / param2, 1), round(param1 / param3, 1), round(param1 / param4, 1), round(param2 / param3, 1),
              round(param2 / param4, 1), round(param3 / param4, 1)]

    if params not in used_params_li:
        # Replace the following functions with your actual implementations
        bigcluster_topicdi, bigcluster_algo_li = search_metric_weight(param1, param2, param3, param4)
        smallcluster_topicdi, smallcluster_scoredf = search_metric_weight_smalltopics(bigcluster_algo_li, param1, param2, param3, param4)
        paramli = [param1, param2, param3, param4]
        paramli = convert_str(paramli)
        pkl_path = '/path/to/your/pkl_files/social_media_clustering_tuples_' + paramli
        output_dir_path = 'path/to/your/csv_output_dir/socialmedia_reddit_' + paramli
        finalcluster_di = eval_small_cluster(smallcluster_topicdi, smallcluster_scoredf)
        sentence_bct, cluster_bct, topic_bct, sentence_table_path, cluster_table_path, topic_table_path, pkl_path = save_results(finalcluster_di, output_dir_path, pkl_path)
        comment_topic_di, topic_cluster_di, topic_statistic_df, topic_di = pickle.load(open(pkl_path, 'rb'))
        sentence_table = add_centroid_dist(comment_topic_di, topic_cluster_di, sentence_table_path, labeled_df)

        if 'eigen_comment_sentiment_score' not in cluster_bct.columns:
            cluster_bct[['eigen_sentence_sentiment_type', 'eigen_sentence_sentiment_score', 'eigen_comment',
                         'eigen_comment_sentiment_type', 'eigen_comment_sentiment_score']] = cluster_bct.apply(
                lambda x: map_sentiment(x.eigen_sentence, sentence_bct, df_full_comments), axis=1, result_type='expand')
            cluster_bct.to_csv(cluster_table_path, index=False)

        topic_table, cluster_table = refine_statistics(sentence_table, cluster_table_path, topic_table_path)
        sentence_table = append_filtered_commsent(topic_statistic_df, df_full_comments, sentence_table, sentence_table_path)
        used_params_li.append(params)
        pickle.dump(obj=used_params_li, file=open('path/to/your/params.pkl', 'wb'))
        clear_output(wait=True)
