# Weekly Keyword Extraction

This Python script, `weekly_keyword_extraction.py`, is designed to extract weekly keywords from Reddit posts using the KeyBERT model. The script connects to a MongoDB database, retrieves Reddit posts from the past week, cleans the text, and then extracts keywords using the KeyBERT model. The extracted keywords are then saved as a pickle file for further analysis.

### Dependencies

- Python 3.6+
- pandas
- numpy
- pymongo
- difflib
- sklearn
- keybert
- keyphrase_vectorizers
- rapidfuzz
- kneed
- tqdm
- pandarallel
- inflect
- html

### How to Use

1. Ensure that all dependencies are installed.
2. Replace the placeholders in the script with your actual MongoDB connection information (e.g., 'your_host', 'your_username', 'your_password', etc.).
3. Run the script using `python weekly_keyword_extraction.py`.

### Functions

- `mongo_to_df(filter_day=7)`: Connects to a MongoDB database, retrieves Reddit posts from the past week, and returns a DataFrame containing the posts.
- `easy_clean_mpnet(text)`: Cleans the input text by removing unwanted characters, links, and email addresses.
- `clean_text(title, selftext, method='mpnet')`: Combines the title and selftext of a Reddit post and cleans the text using the specified method.
- `gb_year_week(col_name, reddit_df=reddit_df)`: Groups the Reddit posts by year and week, and returns a DataFrame with aggregated lists of the specified column.
- `mpnet_kw(liostr)`: Extracts keywords from a list of strings using the KeyBERT model.

### Output

The script saves the extracted keywords as a pickle file with the following naming convention: `mpnetdf_kwdf_YEAR_WEEK.pkl`, where `YEAR` and `WEEK` are the year and week number of the processed data.

### Example

```python
reddit_df = mongo_to_df(12)
reddit_df['doc_mpnet'] = reddit_df.apply(lambda x: clean_text(x.title, x.selftext, method='mpnet'), axis=1)
mpnet_posts_df = gb_year_week('doc_mpnet')
mpnet_posts_df['count'] = mpnet_posts_df['doc_mpnet'].apply(len)

process_year = mpnet_posts_df[mpnet_posts_df['count'] == mpnet_posts_df['count'].max()]['year'].iloc[0]
process_week = mpnet_posts_df[mpnet_posts_df['count'] == mpnet_posts_df['count'].max()]['week'].iloc[0]

mpnet_posts_df = mpnet_posts_df[(mpnet_posts_df['year'] == process_year) & (mpnet_posts_df['week'] == process_week)]
mpnet_posts_df['candidate_keywords'] = mpnet_posts_df['doc_mpnet'].apply(mpnet_kw)

pickle.dump(obj=mpnet_posts_df, file=open('../data/mpnetdf_kwdf_' + str(process_year) + '_' + str(process_week) + '.pkl', 'wb'))
```

This example demonstrates how to use the functions in the script to extract keywords from Reddit posts and save the results as a pickle file.