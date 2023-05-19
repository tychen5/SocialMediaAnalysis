import os
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from transformers import pipeline

# Download all NLTK data
nltk.download('all')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Get the current directory and parameter file path
current_directory = os.path.dirname(__file__)
param_path = os.path.join(current_directory, 'parameter.json')

# Load parameters from the JSON file
with open(param_path, 'r') as f:
    param_dict = json.load(f)

# Get the models path from the parameters
models_path = os.path.join(current_directory, param_dict["models_path"])

def init_clfs():
    """
    Initialize the classifiers using the transformers pipeline.

    Returns:
        tuple: Three classifiers with different models.
    """
    classifier = pipeline("zero-shot-classification",
                          model="facebook/bart-large-mnli",
                          num_workers=os.cpu_count(),
                          device=param_dict["model1_gpu_device"])
    classifier2 = pipeline("zero-shot-classification",
                           model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
                           num_workers=os.cpu_count(),
                           device=param_dict["model2_gpu_device"])
    classifier3 = pipeline("zero-shot-classification",
                           model="joeddav/xlm-roberta-large-xnli",
                           num_workers=os.cpu_count(),
                           device=param_dict["model3_gpu_device"])
    return classifier, classifier2, classifier3

# Initialize the classifiers and save them using pickle
classifier, classifier2, classifier3 = init_clfs()
pickle.dump(obj=(classifier, classifier2, classifier3), file=open(models_path, 'wb'))