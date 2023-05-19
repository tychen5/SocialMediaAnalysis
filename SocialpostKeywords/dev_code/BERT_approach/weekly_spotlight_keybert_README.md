# Weekly Spotlight KeyBERT

`weekly_spotlight_keybert.py` is a Python script that extracts and analyzes data from a MongoDB database containing Reddit posts. The script processes the text data, groups the posts by year and week, and then uses the KeyBERT library to extract keywords from the posts. The extracted keywords can be used for further analysis or to generate weekly summaries of the most discussed topics on Reddit.

### Dependencies

- Python 3.6+
- pandas
- numpy
- pymongo
- difflib
- sklearn
- keybert
- keyphrase_vectorizers
- pandarallel
- inflect
- html
- SentenceTransformer

### Functions

- `mongo_to_df(filter_day=7)`: Connects to a MongoDB database, retrieves Reddit posts created within the specified number of days, and returns a DataFrame containing the data.
- `easy_clean_use(doc)`: Cleans a given text document by removing emails, URLs, and unnecessary whitespace.
- `easy_clean_mpnet(text)`: Cleans a given text document by removing emails, URLs, and unnecessary whitespace.
- `clean_text(title, selftext, method='mpnet')`: Cleans the title and selftext of a Reddit post using the specified method ('mpnet' or 'use').
- `gb_year_week(col_name, reddit_df=reddit_df)`: Groups the Reddit posts by year and week, and returns a DataFrame containing the aggregated data.
- `mpnet_kw(liostr)`: Extracts keywords from a list of text documents using the KeyBERT library.
- `mpnet_kw_multi(liostr)`: Extracts keywords from a list of text documents using the KeyBERT library in a multi-threaded manner.

### Usage

1. Replace the placeholders in the `mongo_to_df` function with your MongoDB connection details.
2. Run the script to retrieve Reddit posts, clean the text data, and group the posts by year and week.
3. Use the `mpnet_kw` or `mpnet_kw_multi` functions to extract keywords from the grouped posts.

### Example

```python
reddit_df = mongo_to_df(365)
reddit_df['doc_mpnet'] = reddit_df.apply(lambda x: clean_text(x.title, x.selftext, method='mpnet'), axis=1)
mpnet_posts_df = gb_year_week('doc_mpnet')
mpnet_sample_postsdf = mpnet_posts_df[-4:]
mpnet_sample_postsdf['candidate_keywords'] = mpnet_sample_postsdf['doc_mpnet'].apply(mpnet_kw_multi)
```

This example retrieves Reddit posts from the past 365 days, cleans the text data, groups the posts by year and week, and extracts keywords from the grouped posts using the KeyBERT library.