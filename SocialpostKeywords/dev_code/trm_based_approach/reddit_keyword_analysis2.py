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

check_point_save_path = '/path/to/your/checkpoint.pkl'

classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli", num_workers=16, device=0)
classifier2 = pipeline("zero-shot-classification",
                       model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", num_workers=16, device=0)
classifier3 = pipeline("zero-shot-classification",
                       model="joeddav/xlm-roberta-large-xnli", num_workers=16, device=1)
classifier4 = pipeline("zero-shot-classification",
                       model="typeform/distilbert-base-uncased-mnli", num_workers=16, device=1)

all_kw_df = []
taken_keywords = []
i = 0


def combine_kw(name, group):
    if type(group) == str:
        return name + ', ' + group
    else:
        return name


while True:
    try:
        df = pd.read_excel('/path/to/your/keywords_list.xlsx', sheet_name=i)
        df.columns = ['keyword', 'similar_words']
        df['model_kw'] = df.apply(lambda x: combine_kw(x.keyword, x.similar_words), axis=1)
        all_kw_df.append(df)
        taken_keywords.extend(df['model_kw'].tolist())
        i = i + 1
    except ValueError:
        break

print(len(taken_keywords))
taken_keywords

# Function to connect to MongoDB and retrieve data
def mongo_to_df(filter_day=7):
    client = pymongo.MongoClient(host='your_host', username='your_username', password='your_password', port=your_port,
                                 authSource='your_authSource', authMechanism='SCRAM-SHA-1')
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


reddit_df = mongo_to_df(365)
reddit_df

bad_puncli = list(set('\|\\|#|-|-|\\-|x200b|\-|\*|>|<|\%|\\\\-|\*|\/|/|*|%|®|tl;dr|x200B|X200b|X200B'.split('|')))
bad_puncli

emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U0001f926-\U0001f937"
                           u"\U00010000-\U0010ffff"
                           u"\u2640-\u2642"
                           u"\u2600-\u2B55"
                           u"\u200d"
                           u"\u23cf"
                           u"\u23e9"
                           u"\u231a"
                           u"\ufe0f"  # dingbats
                           u"\u3030"
                           "]+", flags=re.UNICODE)


def easy_clean_mpnet(text):
    paragraph = html.unescape(text)
    # filter emails
    email = re.search("([^@|\s]+@[^@]+\.[^@|\s]+)", paragraph, re.I)
    try:
        email = email.group(1)
        paragraph = paragraph.replace(email, '')
    except AttributeError:
        pass
    # filter urls
    paragraph = re.sub(r'https?://\S+|www\.\S+', '', paragraph)
    # filter emojis
    paragraph = emoji_pattern.sub(r'', paragraph)
    paragraph = re.sub(r'\n|\r|\t', '', paragraph)
    for punc in bad_puncli:
        paragraph = paragraph.replace(punc, ' ')
        paragraph = paragraph.replace('..', '.')
        paragraph = paragraph.replace('x200b', ' ')
    doc_li = paragraph.split(' ')
    return " ".join(list(filter(None, doc_li)))


def clean_text(title, selftext, method='mpnet'):
    if method == 'mpnet':  # not paragraph
        title = easy_clean_mpnet(title)
        selftext = easy_clean_mpnet(selftext)
        return title + '. ' + selftext
    elif method == 'use':  # paragraphize
        title_li = easy_clean_use(title)
        selftext_li = easy_clean_use(selftext)
        title_li.extend(selftext_li)
        return title_li.copy()
    else:
        print('unknown model!')


reddit_df['full_doc'] = reddit_df.apply(lambda x: clean_text(x.title, x.selftext, method='mpnet'), axis=1)


def easy_clean_use(doc):
    doc = html.unescape(doc)
    # filter emails
    email = re.search("([^@|\s]+@[^@]+\.[^@|\s]+)", doc, re.I)
    try:
        email = email.group(1)
        doc = doc.replace(email, '')
    except AttributeError:
        pass
    # filter urls
    doc = re.sub(r'https?://\S+|www\.\S+', '', doc)
    # filter emojis
    doc = emoji_pattern.sub(r'', doc)
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


reddit_df['paragraph_doc'] = reddit_df.apply(lambda x: clean_text(x.title, x.selftext, method='use'), axis=1)
reddit_df

all_keywords, all_timeli = pickle.load(open(check_point_save_path, 'rb'))
all_timeli
combine_dict = {}
for kw_li, time in zip(all_keywords, all_timeli):
    combine_dict[time] = kw_li
