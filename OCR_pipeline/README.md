# Introduction
1. Using Google Vision API provided by Elton to get OCR results in short posts' images
2. Using grammar check models(CoLA: deberta-v3-large-cola,roberta-large-cola-krishna2020) to extract sentences(paragrpahs) are legal.
3. Using entailment models(MNLI: bart-large-mnli,DeBERTa-v3-large-mnli-fever-anli-ling-wanli) to filter sentences(paragrahs) support the user text typed in the post(title+selftext)

# Usage
1. parameter.json
    1. to set pickle db path
    2. use_device: if cpu=-1, gpu=cuda# e.g., 0,1,2,...
    3. ocr_result_df_path: pickle db path
2. install_models.py
    1. need to run this first before running inference_post_ocr.py
    2. every time with different setting of parameter.json need to re-run this
3. inference_post_ocr.py
    1. main function to get post's ocr result
    2. recommend only run for short posts w/ images