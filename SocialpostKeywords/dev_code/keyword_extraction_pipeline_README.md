# Keyword Extraction Pipeline

This Python script, `keyword_extraction_pipeline.py`, is designed to extract keywords from Reddit posts using two different models: the MiniLM-PyTorch-Net (MPNet) model and the Universal Sentence Encoder (USE) model. The script fetches Reddit posts from a MongoDB database, cleans the text, and then applies the keyword extraction models to generate a list of candidate keywords for each post.

### Dependencies

The script requires the following Python libraries:

- os
- time
- random
- gc
- pandas
- numpy
- pickle
- datetime
- pymongo
- itertools
- re
- string
- html
- inflect
- pandarallel
- difflib
- sklearn
- keybert
- keyphrase_vectorizers
- tensorflow_hub

### Functions

The script contains the following functions:

1. `mongo_to_df(filter_day=7)`: Fetches Reddit posts from a MongoDB database and returns a DataFrame containing the posts.
2. `easy_clean_use(doc)`: Cleans the text for the USE model.
3. `easy_clean_mpnet(text)`: Cleans the text for the MPNet model.
4. `clean_text(title, selftext, method='mpnet')`: Combines and cleans the title and selftext of a Reddit post based on the specified method (MPNet or USE).
5. `gb_year_week(col_name, reddit_df=reddit_df)`: Groups the Reddit posts by year and week.
6. `func1(mpnet_sample_postsdf)`: Extracts keywords using the MPNet model.
7. `func2(use_sample_postsdf)`: Extracts keywords using the USE model.

### Workflow

1. Fetch Reddit posts from the MongoDB database using `mongo_to_df()`.
2. Clean the text of the Reddit posts using `clean_text()` for both MPNet and USE models.
3. Group the Reddit posts by year and week using `gb_year_week()`.
4. Load the Universal Sentence Encoder model from TensorFlow Hub.
5. Extract keywords from the Reddit posts using the MPNet and USE models in parallel using `func1()` and `func2()`.

### Usage

To run the script, simply execute the following command:

```
python keyword_extraction_pipeline.py
```

This will start the keyword extraction process using both the MPNet and USE models in parallel. The extracted keywords will be stored in the respective DataFrames (`mpnet_sample_postsdf` and `use_sample_postsdf`).