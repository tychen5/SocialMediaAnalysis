
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

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def mongo_to_df(filter_day=7):
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


def easy_clean_use(doc):
    doc = html.unescape(doc)
    email = re.search("([^@|\s]+@[^@]+\.[^@|\s]+)", doc, re.I)
    try:
        email = email.group(1)
        doc = doc.replace(email, '')
    except AttributeError:
        pass
    doc = re.sub(r'https?://\S+|www\.\S+', '', doc)
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
    paragraph = html.unescape(text)
    email = re.search("([^@|\s]+@[^@]+\.[^@|\s]+)", paragraph, re.I)
    try:
        email = email.group(1)
        paragraph = paragraph.replace(email, '')
    except AttributeError:
        pass
    paragraph = re.sub(r'https?://\S+|www\.\S+', '', paragraph)
    paragraph = re.sub(r'\n|\r|\t', '', paragraph)
    doc_li = paragraph.split(' ')
    return " ".join(list(filter(None, doc_li)))


def clean_text(title, selftext, method='mpnet'):
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
        print('unknown model!')


reddit_df['doc_mpnet'] = reddit_df.apply(lambda x: clean_text(x.title, x.selftext, method='mpnet'), axis=1)
reddit_df['doc_use'] = reddit_df.apply(lambda x: clean_text(x.title, x.selftext, method='use'), axis=1)


def gb_year_week(col_name, reddit_df=reddit_df):
    take_posts = reddit_df[['year', 'week', col_name]]
    take_df = take_posts.groupby(['year', 'week'], as_index=False).agg(list)
    return take_df


mpnet_posts_df = gb_year_week('doc_mpnet')
use_posts_df = gb_year_week('doc_use')

vectorizer = KeyphraseCountVectorizer(spacy_pipeline='en_core_web_trf')


class CustomEmbedder(BaseEmbedder):
    def __init__(self, embedding_model):
        super().__init__()
        self.embedding_model = embedding_model
        self.pool = self.embedding_model.start_multi_process_pool()

    def embed(self, documents, verbose=False):
        embeddings = self.embedding_model.encode_multi_process(documents, self.pool)
        return embeddings


model = SentenceTransformer("all-mpnet-base-v2")
custom_embedder = CustomEmbedder(embedding_model=model)

mpnet_sample_postsdf = mpnet_posts_df[-4:]


kw_model = KeyBERT(model=custom_embedder)


def mpnet_kw(liostr):
    candidate_kw_li = []
    for doc in liostr:
        key_words = kw_model.extract_keywords(doc, vectorizer=vectorizer, diversity=0.2, use_mmr=True, top_n=10)
        candidate_kw_li.append(key_words)
    return candidate_kw_li


def mpnet_kw_multi(liostr):
    def process(doc):
        key_words = kw_model.extract_keywords(doc, vectorizer=vectorizer, diversity=0.2, use_mmr=True, top_n=10)
        return key_words

    pool = ThreadPool()
    results = pool.map(process, liostr)
    pool.close()
    pool.join()
    return results


mpnet_sample_postsdf['candidate_keywords'] = mpnet_sample_postsdf['doc_mpnet'].apply(mpnet_kw_multi)