# This script initializes two sets of models for text classification and zero-shot classification
# It reads parameters from a JSON file and sets the appropriate directories and device to use
# The models are loaded using the transformers library and are either loaded onto the CPU or GPU depending on the device specified
# The models are then pickled and saved to disk for later use

from transformers import pipeline
import pickle
import os,json
import gc,torch

# Get the current directory and parameter file path
current_directory = os.path.dirname(__file__)
param_path = os.path.join(current_directory, 'parameter.json')

# Load the parameters from the JSON file
with open(param_path,'r') as f:
    param_dict = json.load(f)

# Set the models directory and device to use
models_dir = os.path.join(current_directory, param_dict["models_dir"])
use_device = int(param_dict["use_device"]) #default 1 gpu or cpu=-1

# Initialize the first set of models for text classification
def init_stage1_clf(use_device):
    '''
    cola grammar models
    '''
    if use_device>-1:
        with torch.no_grad():
            grammar1 = pipeline("text-classification", #LABEL_1=accept  , num_worker還可以改
                                model="yevheniimaslov/deberta-v3-large-cola",num_workers=1,device=use_device,truncation=True,max_length=512) 
            grammar2 = pipeline("text-classification", #LABEL_0=accept , num_worker還可以改
                                model="cointegrated/roberta-large-cola-krishna2020",num_workers=1,device=use_device, truncation=True,max_length=512) 
    else:
        grammar1 = pipeline("text-classification", #LABEL_1=accept  , num_worker還可以改
                              model="yevheniimaslov/deberta-v3-large-cola",num_workers=os.cpu_count(),device=-1, truncation=True,max_length=512) 
        grammar2 = pipeline("text-classification", #LABEL_0=accept , num_worker還可以改
                              model="cointegrated/roberta-large-cola-krishna2020",num_workers=os.cpu_count(),device=-1, truncation=True,max_length=512)         
    return grammar1,grammar2

# Initialize the second set of models for zero-shot classification
def init_stage2_clf(use_device):
    '''
    mnli entailment models
    '''
    if use_device>-1:
        with torch.no_grad():
            topic1 = pipeline("zero-shot-classification",
                                    model="facebook/bart-large-mnli",num_workers=1,device=use_device, truncation=True,max_length=1024)  
            topic2 = pipeline("zero-shot-classification",
                                model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",num_workers=1,device=use_device, truncation=True,max_length=512)
    else:
        topic1 = pipeline("zero-shot-classification",
                                  model="facebook/bart-large-mnli",num_workers=os.cpu_count(),device=-1, truncation=True,max_length=1024)  
        topic2 = pipeline("zero-shot-classification",
                              model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",num_workers=os.cpu_count(),device=-1, truncation=True,max_length=512)        
    return topic1,topic2

# Initialize the first set of models and pickle them to disk
grammar1,grammar2 = init_stage1_clf(use_device)
pickle.dump(obj=(grammar1,grammar2),file=open(models_dir+'grammar_model.pkl','wb'))

# Delete the first set of models and free up memory
del grammar1
del grammar2
gc.collect()

# Empty the GPU cache if a GPU is being used
if use_device>-1:
    torch.cuda.empty_cache()  

# Initialize the second set of models and pickle them to disk
topic1,topic2 = init_stage2_clf(use_device)
pickle.dump(obj=topic1,file=open(models_dir+'topic_model1.pkl','wb'))
pickle.dump(obj=topic2,file=open(models_dir+'topic_model2.pkl','wb'))