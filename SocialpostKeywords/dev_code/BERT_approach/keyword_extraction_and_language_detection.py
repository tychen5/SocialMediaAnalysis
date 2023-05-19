#!/usr/bin/env python
# coding: utf-8

# Import required libraries
from keybert import KeyBERT
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector

# Define a sample document
doc = """
         Supervised learning is the machine learning task of learning a function that
         maps an input to an output based on example input-output pairs.[1] It infers a
         function from labeled training data consisting of a set of training examples.[2]
         In supervised learning, each example is a pair consisting of an input object
         (typically a vector) and a desired output value (also called the supervisory signal). 
         A supervised learning algorithm analyzes the training data and produces an inferred function, 
         which can be used for mapping new examples. An optimal scenario will allow for the 
         algorithm to correctly determine the class labels for unseen instances. This requires 
         the learning algorithm to generalize from the training data to unseen situations in a 
         'reasonable' way (see inductive bias).
      """

# Initialize KeyBERT model
kw_model = KeyBERT()

# Extract keywords using MMR
keywords_mmr = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 3),
                                         use_mmr=True, diversity=0.2, stop_words='english')

# Extract keywords using MaxSum
keywords_maxsum = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 3),
                                            use_maxsum=True, nr_candidates=20, top_n=5, stop_words=None)

# Define a function to get language detector
def get_lang_detector(nlp, name):
    return LanguageDetector()

# Initialize Spacy model with language detector
nlp = spacy.load("xx_ent_wiki_sm")
nlp.add_pipe('sentencizer')
nlp.add_pipe('language_detector', last=True)

# Define a sample text
text = '私は知ることを望まない？それで、あなたは知りたいですか？寝たい'

# Process the text with Spacy model
doc = nlp(text)

# Print detected language
print(doc._.language)

# Print sentences in the text
for sent in doc.sents:
    print(sent.text)