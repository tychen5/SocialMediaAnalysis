
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
import tensorflow_hub

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def mongo_to_df(filter_day=7):
    client = pymongo.MongoClient(host='enter_your_host', username='enter_your_username', password='enter_your_password', port=enter_your_port,
                                 authSource='reddit_ui', authMechanism='SCRAM-SHA-1')
    db = client.reddit_ui

    lastweekday = (dt.datetime.today() - dt.timedelta(days=filter_day)).replace(hour=0, minute=0, second=0, microsecond=0)
    reddit_column = ['subreddit', 'author', 'created_utc', 'year','week' ,'full_link', 'link_flair_text', 'title', 'selftext']
    projection = {i: 1 for i in reddit_column}
    projection['_id'] = 0
    prod_collection = db.posts.find({'created_utc': {'$gte': lastweekday}}, projection).sort('created_utc',pymongo.DESCENDING)
    df = pd.DataFrame(prod_collection)

    return df

reddit_df = mongo_to_df(365)

def easy_clean_use(doc):
    doc = html.unescape(doc)
    email = re.search("([^@|\s]+@[^@]+\.[^@|\s]+)",doc,re.I)
    try:
        email = email.group(1)
        doc = doc.replace(email,'')
    except AttributeError:
        pass 
    doc = re.sub(r'https?://\S+|www\.\S+', '', doc)    
    doc_li = re.split(r'\n\n|\n\t',doc)
    clean_li = []
    for paragraph in doc_li:
        paragraph = re.sub(r'\n|\r|\t','',paragraph)
        paragraph_li = paragraph.split(' ')
        paragraph_li = list(filter(None,paragraph_li))
        if len(paragraph_li)>5:
            clean_li.append(" ".join(paragraph_li))
    return clean_li  

def easy_clean_mpnet(text):
    paragraph = html.unescape(text)
    email = re.search("([^@|\s]+@[^@]+\.[^@|\s]+)",paragraph,re.I)
    try:
        email = email.group(1)
        paragraph = paragraph.replace(email,'')
    except AttributeError:
        pass 
    paragraph = re.sub(r'https?://\S+|www\.\S+', '', paragraph)
    paragraph = re.sub(r'\n|\r|\t','',paragraph)
    doc_li = paragraph.split(' ')
    return " ".join(list(filter(None,doc_li)))

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

embedding_model = tensorflow_hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

mpnet_sample_postsdf = mpnet_posts_df[-2:-1]
use_sample_postsdf = use_posts_df[-2:-1]

def func1(mpnet_sample_postsdf):
    def mpnet_kw(liostr):
        kw_model = KeyBERT(model="all-mpnet-base-v2")
        vectorizer = KeyphraseCountVectorizer(spacy_pipeline='en_core_web_trf')
        candidate_kw_li = []
        for doc in tqdm(liostr):
            key_words = kw_model.extract_keywords(doc, vectorizer=vectorizer, diversity=0.8, use_mmr=True, top_n=10)
            candidate_kw_li.append(key_words)
        return candidate_kw_li    
    
    print('start1!')
    mpnet_sample_postsdf['candidate_keywords'] = mpnet_sample_postsdf['doc_mpnet'].apply(mpnet_kw)
    return mpnet_sample_postsdf

def func2(use_sample_postsdf):
    def use_kw(lioliostr):
        liostr = list(itertools.chain.from_iterable(lioliostr))
        liostr = list(filter(None, liostr))
        kw_model = KeyBERT(model=embedding_model)
        vectorizer = KeyphraseCountVectorizer(spacy_pipeline='en_core_web_trf')
        candidate_kw_li = []
        for doc in tqdm(liostr):
            key_words = kw_model.extract_keywords(doc, vectorizer=vectorizer, diversity=0.2, use_mmr=True, top_n=10)
            candidate_kw_li.append(key_words)
        return candidate_kw_li    
    print('start2!')
    use_sample_postsdf['candidate_keywords'] = use_sample_postsdf['doc_use'].apply(use_kw)
    return use_sample_postsdf

if __name__ == '__main__':
    p1 = multiprocessing.Process(target=func1, args=(mpnet_sample_postsdf,))
    p2 = multiprocessing.Process(target=func2, args=(use_sample_postsdf,))
    p1.start()
    p2.start()    
    p1.join()
    p2.join()
