
#!/usr/bin/env python
# coding: utf-8

import os
import pickle
from transformers import pipeline

# Set the directory for the model
current_directory = '/path/to/your/model/directory'

# Set the path for the saved classifiers
models_path = os.path.join(current_directory, 'trm_clfs.pkl')


def init_clfs():
    """
    Initialize the classifiers and return them as a tuple.
    """
    classifier1 = pipeline("zero-shot-classification",
                           model="facebook/bart-large-mnli",
                           num_workers=os.cpu_count())
    classifier2 = pipeline("zero-shot-classification",
                           model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
                           num_workers=os.cpu_count())
    classifier3 = pipeline("zero-shot-classification",
                           model="joeddav/xlm-roberta-large-xnli",
                           num_workers=os.cpu_count())
    classifier4 = pipeline("zero-shot-classification",
                           model="typeform/distilbert-base-uncased-mnli",
                           num_workers=os.cpu_count())
    return classifier1, classifier2, classifier3, classifier4


# Initialize the classifiers
classifiers = init_clfs()

# Save the classifiers to a file
pickle.dump(obj=classifiers, file=open(models_path, 'wb'))
