
#!/usr/bin/env python
# coding: utf-8

# Install keyphrase-vectorizers package
#!pip install keyphrase-vectorizers

# Import necessary libraries
import os
from bertopic import BERTopic
from keybert import KeyBERT
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

# Set environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Example labels for Fowlkes-Mallows score calculation
labels_true = [0, 0, 0, 0, 1, 1, 1, 1]
labels_pred = [0, 0, 0, 1, 1, 1, 1, 1]
metrics.fowlkes_mallows_score(labels_true, labels_pred)

# Initialize KeyBERT model
kw_model = KeyBERT(model="all-mpnet-base-v2")

# Example document
doc = "I'm building an office network for about 150 employees. There will be about 100 desktop computers connected by wire. 30-60 other wired devices like video conference devices and cameras. 200-300 wireless devices. Would the gear below be sufficient or am I missing or having too much something? We don't host anything on site and all the users are quite light weight users just sending emails, making video calls, using their phones, instant messaging."

# Set worker number
worker_num = 13

# Initialize KeyphraseCountVectorizer
vectorizer = KeyphraseCountVectorizer(spacy_pipeline='en_core_web_trf')

# Extract keywords using KeyBERT model
kw_model.extract_keywords(doc, vectorizer=vectorizer, diversity=0.2, use_mmr=True, top_n=10)
