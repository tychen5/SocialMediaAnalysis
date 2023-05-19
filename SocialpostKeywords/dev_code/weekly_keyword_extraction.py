
#!/usr/bin/env python
# coding: utf-8

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
from tqdm.auto import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

bad_puncli = list(set('\|\\|#|-|-|\\-|x200b|\-|\*|>|<|\%|\\\\-|\*|\/|/|*|%|®|tl;dr|x200B|X200b|X200B'.split('|')))


def mongo_to_df(filter_day=7):
    client = pymongo.MongoClient(host='your_host', username='your_username', password='your_password', port=your_port,
                                 authSource='your_authSource', authMechanism='your_authMechanism')
    db = client.reddit_ui

    lastweekday = (dt.datetime.today() - dt.timedelta(days=filter_day)).replace(hour=0, minute=0, second=0, microsecond=0)
    reddit_column = ['subreddit', 'author', 'created_utc', 'year', 'week', 'full_link', 'link_flair_text', 'title',
                     'selftext', 'productline', 'productname']
    projection = {i: 1 for i in reddit_column}
    projection['_id'] = 0
    prod_collection = db.posts.find({'created_utc': {'$gte': lastweekday}}, projection).sort('created_utc',
                                                                                            pymongo.DESCENDING)
    df = pd.DataFrame(prod_collection)

    return df


def easy_clean_mpnet(text):
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
        paragraph = paragraph.replace('x200b', ' ')
    doc_li = paragraph.split(' ')
    return " ".join(list(filter(None, doc_li)))


def clean_text(title, selftext, method='mpnet'):
    if method == 'mpnet':
        title = easy_clean_mpnet(title)
        selftext = easy_clean_mpnet(selftext)
        return title + '. ' + selftext
    else:
        print('unknown model!')


def gb_year_week(col_name, reddit_df=reddit_df):
    take_posts = reddit_df[['year', 'week', col_name]]
    take_df = take_posts.groupby(['year', 'week'], as_index=False).agg(list)
    return take_df


vectorizer = KeyphraseCountVectorizer(spacy_pipeline='en_core_web_trf',
                                      pos_pattern='<J.*>*<N.*>+|<R.*>*<V.*>+<N.*>*|<N.*>+<V.*>+',
                                      )

kw_model = KeyBERT(model="all-mpnet-base-v2")


def mpnet_kw(liostr):
    candidate_kw_li = []
    for doc in tqdm(liostr):
        key_wordsli = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 4), stop_words='english', diversity=0.8,
                                                use_mmr=True, top_n=10)
        key_words = kw_model.extract_keywords(doc, vectorizer=vectorizer, stop_words='english', diversity=0.7,
                                              use_mmr=True, top_n=10)
        key_wordsli.extend(key_words)
        candidate_kw_li.append(key_wordsli)
    return candidate_kw_li


reddit_df = mongo_to_df(12)
reddit_df['doc_mpnet'] = reddit_df.apply(lambda x: clean_text(x.title, x.selftext, method='mpnet'), axis=1)
mpnet_posts_df = gb_year_week('doc_mpnet')
mpnet_posts_df['count'] = mpnet_posts_df['doc_mpnet'].apply(len)

process_year = mpnet_posts_df[mpnet_posts_df['count'] == mpnet_posts_df['count'].max()]['year'].iloc[0]
process_week = mpnet_posts_df[mpnet_posts_df['count'] == mpnet_posts_df['count'].max()]['week'].iloc[0]

mpnet_posts_df = mpnet_posts_df[(mpnet_posts_df['year'] == process_year) & (mpnet_posts_df['week'] == process_week)]
mpnet_posts_df['candidate_keywords'] = mpnet_posts_df['doc_mpnet'].apply(mpnet_kw)

pickle.dump(obj=mpnet_posts_df, file=open('../data/mpnetdf_kwdf_' + str(process_year) + '_' + str(process_week) + '.pkl', 'wb'))