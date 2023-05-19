Installation:
May need 16GB of cpu RAM or more.
===Step 1===
Python => 3.8.13 or above + transformers => 4.20.1
or
Docker env => huggingface/transformers-all-latest-gpu
===Step 2===
python install_models.py (will automatically save classifiers into pickles to speed up while inferencing)


Inferencing:
===Input (in order):===
original reddit title: e.g., 'Multiple LAN; select WAN?'
original reddit body: e.g., 'I have a UDM-SE and currently have two WAN pro'
original reddit id: e.g., 'wcznq3'
Tips: if want to re-inference an old post, add a diggit to id.
===Output (may be empty):===
list of keywords: e.g., ['udm']

Release notes:
v2.0: update output format, output category level name
1. update all configs to json file (if change to use gpu or not, need to re-run python install_models.py) 
    * use gpu: set gpu_device=0 (gpu id)
    * not use gpu: set gpu_device=-1
v3.0: update keywords to 3-level topic
1. update result to 3-level structure
2. return dict of topic, possible product line
v4.0: update title rule base topic
1. need to install nltk and download it's corpus