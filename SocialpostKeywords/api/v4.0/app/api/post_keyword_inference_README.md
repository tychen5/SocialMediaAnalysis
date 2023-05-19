`post_keyword_inference.py`

# README.md

## Post Keyword Inference

This Python script, `post_keyword_inference.py`, contains two main functions that perform keyword and topic inference on a given post using in-house models. The script is designed to process and analyze the title and body of a post, and then return a list of relevant keywords and topics.

### Functions

#### 1. `title_em_rule_inference()`

This function performs rule inference on the title of a post using the given classifiers and dictionaries. It takes in the preprocessed title, original title, taken keywords, list of keywords in the title, two classifier functions, and three dictionaries as input arguments. The function returns a sorted list of inferred topics.

#### 2. `post_keywords_inference_inhouse()`

This function performs keyword inference on a post using in-house models. It takes in the post's original title, body, and id as input arguments. The function returns a list of predefined keywords in the dictionary.

### Usage

To use the `post_keyword_inference.py` script, simply import the functions and call them with the appropriate input arguments. For example:

```python
from post_keyword_inference import post_keywords_inference_inhouse

title = "Sample Post Title"
selftext = "This is the body of the sample post."
id_key = "sample_post_id"

keywords = post_keywords_inference_inhouse(title, selftext, id_key)
print(keywords)
```

### Dependencies

The script requires the following libraries:

- pandas
- functools
- nltk

Additionally, the script requires access to the following files:

- `/path/to/your/keyword_dict`: A pickled file containing keyword dictionaries.
- `/path/to/your/models`: A pickled file containing the classifier models.

Please replace `/path/to/your/keyword_dict` and `/path/to/your/models` with the actual paths to your files.

### Note

This script assumes that the necessary preprocessing and cleaning functions, such as `clean_text()`, `em_preprocess_algo()`, `long_post_model_inference()`, `short_post_model_inference()`, and `get_kw_name()` are available in the same module or have been imported from another module.