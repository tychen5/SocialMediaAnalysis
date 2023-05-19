
import os
import time
import random
import gc
import pandas as pd
import numpy as np
import pickle
import datetime as dt
import pymongo
import itertools
import re
import string
import html
import inflect
from pandarallel import pandarallel
import difflib
from sklearn import metrics
from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
from rapidfuzz import fuzz
from kneed import KneeLocator
from difflib import SequenceMatcher
import operator
import tensorflow_hub

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def mongo_to_df(filter_day=7):
    """
    Connect to MongoDB and retrieve data as a DataFrame.
    """
    client = pymongo.MongoClient(host='your_host', username='your_username', password='your_password', port=your_port,
                                 authSource='your_authSource', authMechanism='your_authMechanism')
    db = client.reddit_ui
    lastweekday = (dt.datetime.today() - dt.timedelta(days=filter_day)).replace(hour=0, minute=0, second=0, microsecond=0)
    reddit_column = ['subreddit', 'author', 'created_utc', 'year', 'week', 'full_link', 'link_flair_text', 'title', 'selftext']
    projection = {i: 1 for i in reddit_column}
    projection['_id'] = 0
    prod_collection = db.posts.find({'created_utc': {'$gte': lastweekday}}, projection).sort('created_utc', pymongo.DESCENDING)
    df = pd.DataFrame(prod_collection)
    return df


reddit_df = mongo_to_df(365)
bad_puncli = list(set('\|\\|#|-|-|\\-|x200b|\-|\*|>|<|\%|\\\\-|\*|\/|/|*|%|®|tl;dr'.split('|')))


def easy_clean_use(doc):
    """
    Clean the document for use with the USE model.
    """
    doc = html.unescape(doc)
    email = re.search("([^@|\s]+@[^@]+\.[^@|\s]+)", doc, re.I)
    try:
        email = email.group(1)
        doc = doc.replace(email, '')
    except AttributeError:
        pass
    doc = re.sub(r'https?://\S+|www\.\S+', '', doc)
    for punc in bad_puncli:
        doc = doc.replace(punc, ' ')
        doc = doc.replace('..', '.')
    doc_li = re.split(r'\n\n|\n\t', doc)
    clean_li = []
    for paragraph in doc_li:
        paragraph = re.sub(r'\n|\r|\t', '', paragraph)
        paragraph_li = paragraph.split(' ')
        paragraph_li = list(filter(None, paragraph_li))
        if len(paragraph_li) > 5:
            clean_li.append(" ".join(paragraph_li))
    return clean_li


def easy_clean_mpnet(text):
    """
    Clean the text for use with the MPNet model.
    """
    paragraph = html.unescape(text)
    email = re.search("([^@|\s]+@[^@]+\.[^@|\s]+)", paragraph, re.I)
    try:
        email = email.group(1)
        paragraph = paragraph.replace(email, '')
    except AttributeError:
        pass
    paragraph = re.sub(r'https?://\S+|www\.\S+', '', paragraph)
    paragraph = re.sub(r'\n|\r|\t', '', paragraph)
    for punc in bad_puncli:
        paragraph = paragraph.replace(punc, ' ')
        paragraph = paragraph.replace('..', '.')
    doc_li = paragraph.split(' ')
    return " ".join(list(filter(None, doc_li)))


def clean_text(title, selftext, method='mpnet'):
    """
    Clean the text based on the specified method.
    """
    if method == 'mpnet':
        title = easy_clean_mpnet(title)
        selftext = easy_clean_mpnet(selftext)
        return title + '. ' + selftext
    elif method == 'use':
        title_li = easy_clean_use(title)
        selftext_li = easy_clean_use(selftext)
        title_li.extend(selftext_li)
        return title_li.copy()
    else:
        pass


reddit_df['doc_mpnet'] = reddit_df.apply(lambda x: clean_text(x.title, x.selftext, method='mpnet'), axis=1)
reddit_df['doc_use'] = reddit_df.apply(lambda x: clean_text(x.title, x.selftext, method='use'), axis=1)
latest_year, latest_week = reddit_df.loc[0, 'year'], reddit_df.loc[0, 'week']


def gb_year_week(col_name, reddit_df=reddit_df):
    """
    Group by year and week.
    """
    take_posts = reddit_df[['year', 'week', col_name]]
    take_df = take_posts.groupby(['year', 'week'], as_index=False).agg(list)
    bad_idx = take_df[(take_df['year'] == latest_year) & (take_df['week'] == latest_week)].index
    take_df = take_df.drop(bad_idx)
    return take_df


mpnet_posts_df = gb_year_week('doc_mpnet')
use_posts_df = gb_year_week('doc_use')
vectorizer = KeyphraseCountVectorizer(spacy_pipeline='en_core_web_trf')
embedding_model = tensorflow_hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
kw_model = KeyBERT(model=embedding_model)
use_sample_postsdf = use_posts_df
mpnet_sample_postsdf = mpnet_posts_df

import numpy as np
import math
from fuzzywuzzy import fuzz
from kneed import KneeLocator
import itertools

def df_tf_score(kw_li, doc_li):
    """
    Calculate document frequency and term frequency scores for a list of keywords.

    Args:
        kw_li (list): List of keywords.
        doc_li (list): List of documents.

    Returns:
        tuple: Mean document frequency score and mean term frequency score.
    """
    df_score_li = []
    tf_score_li = []

    for kw in kw_li:
        df_count = 0
        tf_score_kw = 0

        for doc in doc_li:
            doc = doc.lower()
            tsr = fuzz.token_set_ratio(kw, doc)
            pr = fuzz.partial_ratio(doc, kw)
            df_score = np.mean([tsr, pr])

            if df_score > 80:
                df_count += 1

            tf_score = np.mean([fuzz.token_sort_ratio(kw, doc), tsr, fuzz.ratio(doc, kw), pr])

            if tf_score > 20:
                tf_score_kw = tf_score_kw + tf_score * 0.01

        df_score_li.append(df_count)
        tf_score_li.append(np.log1p(tf_score_kw))

    return np.mean(df_score_li), np.mean(tf_score_li)

# Replace 'take_kw' with your actual DataFrame
take_kw[['df', 'tf']] = take_kw.apply(lambda x: df_tf_score(x.kw_group), axis=1, result_type='expand')
take_kw['kw_score'] = take_kw['df'] * take_kw['tf']
take_kw = take_kw.sort_values(['kw_score'], ascending=False)

# Replace 'y' with your actual list of keyword scores
y = sorted(y, reverse=False)
x = [i for i in range(len(y))]
kneedle = KneeLocator(x, y, curve="convex", direction="increasing", online=False, S=3)
score_thr = kneedle.knee_y

# Replace 'overall_kw_df' with your actual DataFrame
overall_kw_df = take_kw[take_kw['kw_score'] > score_thr]
overall_kw_df = overall_kw_df[['kw_name', 'kw_group', 'df', 'tf', 'kw_score']]

output_di = {}
for idx, row in overall_kw_df.iterrows():
    all_kwli = row['kw_group'].copy()
    all_kwli.extend(row['kw_name'].split('/'))
    final_kw = sorted(set(all_kwli), key=len)
    kw_name = "|".join(final_kw)
    output_di[kw_name] = {}
    output_di[kw_name]['doc_freq'] = math.ceil(row['df'])
    output_di[kw_name]['kw_popularity'] = row['kw_score']
