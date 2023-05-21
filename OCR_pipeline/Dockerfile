# base on huggingface
FROM huggingface/transformers-all-latest-gpu
# install script
RUN pip install pandas==1.3.5 transformers[sentencepiece]==4.20.1 openpyxl thefuzz[speedup] nvidia-ml-py3 ray=2.0.0
RUN python ./api/install_models.py 