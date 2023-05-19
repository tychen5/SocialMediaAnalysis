
#!/usr/bin/env python
# coding: utf-8

import os
import time
import random
import gc
import itertools
import re
import string
import html
import inflect
import operator
import datetime as dt
import pymongo
import pandas as pd
import numpy as np
import pickle
from pandarallel import pandarallel
from sklearn import metrics
from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
from rapidfuzz import fuzz
from kneed import KneeLocator
from difflib import SequenceMatcher
from tqdm.auto import tqdm
import tensorflow_hub

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

bad_puncli = list(set('\|\\|#|-|-|\\-|x200b|\-|\*|>|<|\%|\\\\-|\*|\/|/|*|%|®|tl;dr|x200B|X200b|X200B'.split('|')))

def mongo_to_df(filter_day=7):
    client = pymongo.MongoClient(host='enter_host', username='enter_username', password='enter_password', port=enter_port,
                                 authSource='enter_authSource', authMechanism='enter_authMechanism')
    db = client.enter_db_name

    lastweekday = (dt.datetime.today() - dt.timedelta(days=filter_day)).replace(hour=0, minute=0, second=0, microsecond=0)
    reddit_column = ['subreddit', 'author', 'created_utc', 'year','week' ,'full_link', 'link_flair_text', 'title', 'selftext','productline','productname']
    projection = {i: 1 for i in reddit_column}
    projection['_id'] = 0
    prod_collection = db.posts.find({'created_utc': {'$gte': lastweekday}}, projection).sort('created_utc',pymongo.DESCENDING)
    df = pd.DataFrame(prod_collection)

    return df

def easy_clean_use(doc):
    doc = html.unescape(doc)
    email = re.search("([^@|\s]+@[^@]+\.[^@|\s]+)",doc,re.I)
    try:
        email = email.group(1)
        doc = doc.replace(email,'')
    except AttributeError:
        pass 
    doc = re.sub(r'https?://\S+|www\.\S+', '', doc)    
    for punc in bad_puncli:
        doc = doc.replace(punc,' ')
        doc = doc.replace('..','.')
    doc_li = re.split(r'\n\n|\n\t',doc)
    clean_li = []
    for paragraph in doc_li:
        paragraph = re.sub(r'\n|\r|\t','',paragraph)
        paragraph_li = paragraph.split(' ')
        paragraph_li = list(filter(None,paragraph_li))
        if len(paragraph_li)>5:
            clean_li.append(" ".join(paragraph_li))
    return clean_li 

def clean_text(title, selftext, method='mpnet'):
    if method=='mpnet':
        title = easy_clean_mpnet(title)
        selftext = easy_clean_mpnet(selftext)
        return title+'. '+selftext
    elif method=='use':
        title_li = easy_clean_use(title)
        selftext_li = easy_clean_use(selftext)
        title_li.extend(selftext_li)
        return title_li.copy()
    else:
        print('unknown model!')

def gb_year_week(col_name, reddit_df=reddit_df):
    take_posts = reddit_df[['year','week',col_name]]
    take_df = take_posts.groupby(['year','week'],as_index=False).agg(list)
    return take_df

def use_kw(lioliostr):
    liostr = list(itertools.chain.from_iterable(lioliostr))
    liostr = list(filter(None,liostr))
    candidate_kw_li = []
    for doc in tqdm(liostr):
        key_wordsli = kw_model.extract_keywords(doc,vectorizer=vectorizer,diversity=0.2,use_mmr=True,top_n=10)
        key_words = kw_model.extract_keywords(doc,keyphrase_ngram_range=(1, 4),diversity=0.7,use_mmr=True,top_n=10)
        key_wordsli.extend(key_words)
        candidate_kw_li.append(key_wordsli)
    return candidate_kw_li

reddit_df = mongo_to_df(12)
reddit_df['doc_use'] = reddit_df.apply(lambda x: clean_text(x.title, x.selftext, method='use'), axis=1)
use_posts_df = gb_year_week('doc_use')
use_posts_df['count'] = use_posts_df['doc_use'].apply(len)

process_year = use_posts_df[use_posts_df['count']==use_posts_df['count'].max()]['year'].iloc[0]
process_week = use_posts_df[use_posts_df['count']==use_posts_df['count'].max()]['week'].iloc[0]
use_posts_df = use_posts_df[(use_posts_df['year']==process_year)&(use_posts_df['week']==process_week)]

vectorizer = KeyphraseCountVectorizer(spacy_pipeline='en_core_web_trf',
                                      pos_pattern='<J.*>*<N.*>+|<R.*>*<V.*>+<N.*>*|<N.*>+<V.*>+',
                                      )
embedding_model = tensorflow_hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
kw_model = KeyBERT(model=embedding_model)

use_posts_df['candidate_keywords'] = use_posts_df['doc_use'].apply(use_kw)

pickle.dump(obj=use_posts_df, file=open('/path/to/your/data/usedf_kwdf_'+str(process_year)+'_'+str(process_week)+'.pkl', 'wb'))