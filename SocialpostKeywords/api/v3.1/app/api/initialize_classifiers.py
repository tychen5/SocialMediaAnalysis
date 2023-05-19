import os
import json
import pickle
from transformers import pipeline

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
    Initialize classifiers using the Hugging Face Transformers library.
    
    Returns:
        tuple: A tuple containing three zero-shot-classification pipelines.
    """
    classifier1 = pipeline("zero-shot-classification",
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

    return classifier1, classifier2, classifier3

# Initialize classifiers and save them using pickle
classifier1, classifier2, classifier3 = init_clfs()
pickle.dump(obj=(classifier1, classifier2, classifier3), file=open(models_path, 'wb'))