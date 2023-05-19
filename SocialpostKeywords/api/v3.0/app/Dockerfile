# base on huggingface
FROM huggingface/transformers-all-latest-gpu
# install script
RUN pip install pandas==1.3.5 transformers[sentencepiece] openpyxl
RUN python ./api/install_models.py 