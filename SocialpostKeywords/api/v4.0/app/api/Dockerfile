# base on huggingface
FROM huggingface/transformers-all-latest-gpu
# install script
RUN pip install pandas==1.3.5 nltk==3.7 transformers[sentencepiece] openpyxl thefuzz[speedup]
RUN python ./api/install_models.py 