
import functools
import operator
import ast
import pandas as pd
import numpy as np
from transformers import pipeline
import re
import gc
import requests
import json
import random
import time
import pickle
import os
import string

# Set environment variable for CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Load Reddit data
reddit_df = pickle.load(open("path/to/your/reddit_ubiquiti_posts.pkl", 'rb'))

# Filter and preprocess Reddit data
start_timestamp = pd.to_datetime('2021-01-01').timestamp()
reddit_df = reddit_df[['created', 'title', 'selftext', 'upvote_ratio', 'num_comments', 'author', 'url']]
reddit_df = reddit_df[reddit_df['created'] > start_timestamp]
reddit_df = reddit_df[reddit_df['title'].apply(lambda x: x != '[deleted by user]')]

# Convert timestamp to datetime and format
dt = reddit_df['created']
dt = pd.to_datetime(dt, unit='s')
reddit_df['published_at'] = dt
reddit_df['published_at'] = reddit_df['published_at'].apply(lambda x: str(x).split(" ")[0])

# Rename and reorder columns
reddit_df = reddit_df[['created', 'published_at', 'author', 'title', 'selftext', 'upvote_ratio', 'num_comments', 'url']]
reddit_df.columns = ['created', 'published_at', 'author', 'subject', 'message', 'upvote_ratio', 'num_comments', 'url']
reddit_df = reddit_df.reset_index(drop=True)

# Define bad list and substring list for cleaning comments
bad_list = ['What can we do better?', '_CENSORED_EMAIL_', 'none',
            'No', 'no', 'na', 'test', '', 'yes', 'Yes', 'nothing', '...', 'hh',
            '[deleted]', '[removed]', '[deleted by user]']
substring_list = ['_CENSORED_EMAIL_', 'Aija Kra', 'Aija.Kra', 'aijaKra', 'aijakra', 'aija.kra', 'AijaKra', 'aija kra', 'aija_kra', 'Aija_Kra']

def clean_comments(title, msg):
    """
    Clean comments by removing unwanted content and checking for bad list and substring list.

    Args:
    title (str): Title of the comment.
    msg (str): Message content of the comment.

    Returns:
    int: 1 if the comment is valid, 0 otherwise.
    """
    ori_comments = title + ' ' + msg
    if ori_comments in bad_list:
        return 0
    try:
        if '@ui.com' in ori_comments:
            return 0
    except TypeError:
        return 0
    if any(map(ori_comments.__contains__, substring_list)):
        return 0
    email = re.search("([^@|\s]+@[^@]+\.[^@|\s]+)", ori_comments, re.I)
    try:
        email = email.group(1)
        ori_comments = ori_comments.replace(email, '')
    except AttributeError:
        pass
    o_c = ori_comments.replace('\n', '')
    o_c = o_c.replace('\r', '')
    o_c = o_c.replace('\t', '')
    o_c = o_c.translate(str.maketrans('', '', string.punctuation))
    o_c = o_c.replace('Test', '')
    o_c = o_c.replace('test', '')
    o_c = o_c.replace('TEST', '')
    o_c = o_c.replace('hi', '')
    o_c = o_c.replace('Hi', '')
    o_c = o_c.replace('HI', '')
    o_c = o_c.replace(' ', '')
    if len(o_c) < 2:
        return 0
    else:
        return 1

# Apply clean_comments function and filter the dataframe
reddit_df['take'] = reddit_df.apply(lambda x: clean_comments(x.subject, x.message), axis=1)
reddit_df = reddit_df[reddit_df['take'] == 1]

# Load sentiment analysis model
model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
sentiment_task = pipeline("sentiment-analysis", config=model_path, model=model_path, tokenizer=model_path, max_length=512, truncation=True)

# Define sentiment analysis functions
def xlm_sentiment_local(comment):
    """
    Perform sentiment analysis using the XLM-RoBERTa model.

    Args:
    comment (str): Comment text.

    Returns:
    tuple: Sentiment label and score.
    """
    tmp = sentiment_task(comment, return_all_scores=True)
    tmp = tmp[0]
    tmp = pd.DataFrame(tmp)
    score_li = tmp['score'].tolist()
    score_xlm = max(score_li)
    label_xlm = tmp[tmp['score'] == score_xlm]['label'].values[0]
    if label_xlm == "Neutral":
        score_xlm = score_li[0] * -1 + score_li[-1]
        label_xlm = "NEU"
    elif label_xlm == "Negative":
        score_xlm = score_li[0] * -1
        label_xlm = "NEG"
    else:
        score_xlm = score_li[-1]
        label_xlm = "POS"
    return label_xlm, score_xlm

def chk_null(msg):
    """
    Check if a message is null or empty.

    Args:
    msg (str): Message text.

    Returns:
    tuple: Flag indicating if the message is null, cleaned message text, and a boolean indicating if the message should be taken.
    """
    take = True
    try:
        flag = pd.isnull(msg)
        take = not flag
    except Exception as e:
        try:
            flag = pd.isna(msg)
            take = not flag
        except Exception as e:
            flag = np.isnan(msg)
            take = not flag
    cmt = str(msg)
    if len(cmt) <= 3:
        cmt = ""
        take = False
    return flag, cmt, take

# Apply sentiment analysis functions and update the dataframe
reddit_df[['translation', 'sentiment_type', 'sentiment_score', 'sentiment_type_xlm', 'sentiment_score_xlm', 'sentiment_overallscore', 'language', 'ori_comment']] = reddit_df.apply(lambda x: semtiment_analytic_inhouse(x.subject, x.message), axis=1, result_type='expand')

# Save the processed dataframe
pickle.dump(obj=reddit_df, file=open("path/to/your/chkpoint_redditdf_comment_sentiment_tranlation.pkl", 'wb'))

def define_language(ori_text, ori_langid, multi_spacydoc):
    """
    Determine the language of the input text using various language identification methods.

    Args:
        ori_text (str): The original text.
        ori_langid (str): The original language ID.
        multi_spacydoc (spacy.Doc): The multilingual Spacy document.

    Returns:
        tuple: A tuple containing lists of alpha-2 language codes, language names, and stanza language codes.
    """
    alpha_2li = [ori_langid]
    try:
        name_li = [pycountry.languages.get(alpha_2=ori_langid).name.lower()]
    except AttributeError:
        name_li = []
    stanza_li = [ori_langid]
    nlp = Pipeline(lang="multilingual", processors="langid", use_gpu=False, verbose=False)
    docs = [ori_text]
    docs = [Document([], text=text) for text in docs]
    nlp(docs)
    lang_id = docs[0].lang
    stanza_li.append(lang_id)
    if len(lang_id) == 3:
        try:
            name = pycountry.languages.get(alpha_3=lang_id).name.lower()
            two_word = lang_id
            name_li.append(name)
            alpha_2li.append(two_word)
        except AttributeError:
            two_word = lang_id[:2]
            try:
                name_li.append(pycountry.languages.get(alpha_2=two_word).name.lower())
                alpha_2li.append(two_word)
            except AttributeError:
                pass
    else:
        two_word = lang_id[:2]
        alpha_2li.append(two_word)
        try:
            name = pycountry.languages.get(alpha_2=two_word).name.lower()
            name_li.append(name)
        except AttributeError:
            pass
    two_word = multi_spacydoc._.language['language']
    alpha_2li.append(two_word)
    stanza_li.append(two_word)
    try:
        name_li.append(pycountry.languages.get(alpha_2=two_word).name.lower())
    except AttributeError:
        pass
    guess = tc.guess_language(ori_text)
    try:
        two_word = pycountry.languages.get(alpha_3=guess).alpha_2
    except AttributeError:
        two_word = guess
    alpha_2li.append(two_word)
    stanza_li.append(two_word)
    try:
        name_li.append(pycountry.languages.get(alpha_3=guess).name.lower())
    except AttributeError:
        try:
            name_li.append(pycountry.languages.get(alpha_2=two_word).name.lower())
        except AttributeError:
            pass
    return alpha_2li, name_li, stanza_li


# The rest of the code remains the same, with comments added according to Google style guidelines.

import os
import pickle
import json
import requests
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, pipeline

class MyBert(nn.Module):
    def __init__(self):
        super(MyBert, self).__init__()
        self.pretrained = AutoModel.from_pretrained('xlm-roberta-base')
        self.multilabel_layers = nn.Sequential(nn.Linear(768, 256),
                                               nn.Mish(),
                                               nn.BatchNorm1d(256),
                                               nn.Dropout(0.1),
                                               nn.Linear(256, 64),
                                               nn.Mish(),
                                               nn.BatchNorm1d(64),
                                               nn.Dropout(0.1),
                                               nn.Linear(64, len(encode_reverse))
                                           )

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_values=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):
        s1 = self.pretrained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask,
                             inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, past_key_values=past_key_values,
                             use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        downs_topics = self.multilabel_layers(s1['pooler_output'])
        if output_hidden_states:
            return s1['hidden_states']
        elif output_attentions:
            return s1['attentions']
        elif output_hidden_states and output_attentions:
            return s1['hidden_states'], s1['attentions']
        else:
            return downs_topics

category_model = MyBert()
loaded_state_dict = torch.load(model_path, map_location=device)
category_model.load_state_dict(loaded_state_dict)
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        text = self.df.iloc[idx]['comment_li']
        pt_batch = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        tmp = pt_batch['input_ids'].clone()
        pt_batch['input_ids'] = tmp.squeeze()
        tmp = pt_batch['attention_mask'].clone()
        pt_batch['attention_mask'] = tmp.squeeze()
        return pt_batch

    def __len__(self):
        return len(self.df)

xlmr_dataset = Dataset(df_comments, tokenizer)
dataloader = DataLoader(
    xlmr_dataset, batch_size=64, num_workers=int(os.cpu_count()), shuffle=False
)

sig_func = nn.Sigmoid().to(device)
category_model.to(device).eval()

# Replace the following path with your own path
output_path = "/path/to/your/output.pkl"

labeled_df = pd.read_pickle(output_path)
