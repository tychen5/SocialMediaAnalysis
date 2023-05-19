import pandas as pd
import numpy as np
from transformers import pipeline
from tqdm.auto import tqdm
import pymongo
import datetime as dt
import html
import re
import pickle
from functools import reduce
import os
import time
import random
import urllib.request
from nltk.stem import WordNetLemmatizer
import nltk
import string
from nltk.corpus import wordnet
from thefuzz import fuzz

check_point_save_path = '/path/to/your/checkpoint.pkl'
check_point_read_path = check_point_save_path
result_save_path = '/path/to/your/results.pkl'
kw_dict_path = '/path/to/your/dictionary.xlsx'
kw_dict_inference_path = '/path/to/your/inference.pkl'
download_thumbnail_path = '/path/to/your/thumbnail.jpg'

classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli", num_workers=os.cpu_count(), device=1)
classifier2 = pipeline("zero-shot-classification",
                       model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", num_workers=os.cpu_count(), device=1)
classifier3 = pipeline("zero-shot-classification",
                       model="joeddav/xlm-roberta-large-xnli", num_workers=os.cpu_count(), device=0)

lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

xls = pd.ExcelFile(kw_dict_path)

def mongo_to_df(filter_day=7):
    client = pymongo.MongoClient(host='enter_your_host', username='enter_your_username', password='enter_your_password', port=enter_your_port,
                                 authSource='reddit_ui', authMechanism='SCRAM-SHA-1')
    db = client.reddit_ui
    lastweekday = (dt.datetime.today() - dt.timedelta(days=filter_day)).replace(hour=0, minute=0, second=0, microsecond=0)

    # ... (rest of the code remains the same)

def get_recalc_time(only_new=False):
    """
    only_new: 如果只有要疊加新的沒有更新word dictionary那就指定為True，否則可能會跑至少三個月的時間，會需要一天多
    """
    try:
        combine_dict, titletopic_dict = pickle.load(open(check_point_read_path, 'rb'))
    except FileNotFoundError:
        combine_dict = {}
        titletopic_dict = {}


def easy_clean_mpnet(text):
    paragraph = html.unescape(text)
    paragraph = re.sub("[{].*[}]", "", paragraph)
    email = re.search("([^@|\s]+@[^@]+\.[^@|\s]+)", paragraph, re.I)

    try:
        email = email.group(1)
        paragraph = paragraph.replace(email, ' ')
    except AttributeError:
        pass


def clean_text(title, selftext, method='mpnet'):
    """
    Clean and combine title and selftext based on the specified method.

    Args:
        title (str): The title of the post.
        selftext (str): The selftext of the post.
        method (str, optional): The cleaning method to use. Defaults to 'mpnet'.

    Returns:
        str: The cleaned and combined text.
    """
    if method == 'mpnet':
        title = easy_clean_mpnet(title)
        selftext = easy_clean_mpnet(selftext)
        if len(title) > 1 and title[-1] in end_punc:
            return title + ' ' + selftext
        else:
            if len(title) > 1:
                return title + '. ' + selftext
            else:
                return selftext
    elif method == 'use':
        title_li = easy_clean_use(title)
        selftext_li = easy_clean_use(selftext)
        title_li.extend(selftext_li)
        return title_li.copy()
    else:
        raise ValueError("Invalid method. Choose 'mpnet' or 'use'.")


reddit_df['full_doc'] = reddit_df.apply(lambda x: clean_text(x.title, x.selftext, method='mpnet'), axis=1)


def easy_clean_use(doc):
    """
    Clean the input document by removing unwanted characters and splitting it into paragraphs.

    Args:
        doc (str): The input document.

    Returns:
        list: A list of cleaned paragraphs.
    """
    doc = html.unescape(doc)
    email = re.search("([^@|\s]+@[^@]+\.[^@|\s]+)", doc, re.I)
    try:
        email = email.group(1)
        doc = doc.replace(email, ' ')
    except AttributeError:
        pass

    doc = re.sub(r'https?://\S+|www\.\S+', ' ', doc)
    doc = emoji_pattern.sub(r'', doc)

    for punc in bad_puncli:
        doc = doc.replace(punc, ' ')
        doc = doc.replace('..', '.')

    doc_li = re.split(r'\n\n|\n\t', doc)
    clean_li = []

    for paragraph in doc_li:
        paragraph = re.sub(r'\n|\r|\t', ' ', paragraph)
        paragraph_li = paragraph.split(' ')
        paragraph_li = list(filter(None, paragraph_li))

        if len(paragraph_li) > 5:
            clean_li.append(" ".join(paragraph_li))

    return clean_li


reddit_df['paragraph_doc'] = reddit_df.apply(lambda x: clean_text(x.title, x.selftext, method='use'), axis=1)
reddit_df['title_emtext'] = reddit_df['title'].apply(em_preprocess_algo)

all_full_doc = reddit_df['full_doc'].tolist()
all_paragraphs = reddit_df['paragraph_doc'].tolist()
all_time = reddit_df['created_utc'].tolist()
all_titles = reddit_df['title_emtext'].tolist()
all_ori_title = reddit_df['title'].tolist()
all_subgroup_li = reddit_df['subreddit'].tolist()

def find_word(liostr, word):
    """Find a word in a list of strings.

    Args:
        liostr (list): List of strings to search in.
        word (str): Word to search for.

    Returns:
        bool: True if the word is found in any of the strings, False otherwise.
    """
    for text in liostr:
        if word in text:
            return True
    return False


# Example usage of find_word function
reddit_df_ori[reddit_df_ori['keyword_li'].apply(find_word, args=('camera/doorbell/protect disconnect',))]

# Accessing specific elements in the DataFrame
reddit_df_ori.loc[13756, 'keyword_li']
reddit_df_ori[reddit_df_ori['date'] == '2022-07-31'].loc[1758, 'keyword_li']

# Example of calculating the mean of a list of numbers
mean_value = np.mean([0.002, 0.0, 0.008, 0.986])

# Example of filtering a DataFrame based on a target issue
idx = 11
target_issue = 'cybersecurity vulnerability issues'
tmp = reddit_df_ori[reddit_df_ori['keyword_li'].apply(lambda x: target_issue in x)]

# Example of accessing a specific location in the DataFrame
loc_idx = 222

# Example list of issues
issues = ['cybersecurity vulnerability issues', 'dhcp issue', 'feature request', 'network topology', 'setup help', 'setup load balance failover', 'vlan question', 'wireless access point overload']

# Example of checking the version of a library
import sentencepiece
print(sentencepiece.__version__)