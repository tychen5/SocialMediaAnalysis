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
