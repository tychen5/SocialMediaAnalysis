 Keyword Extraction Pipeline

This Python script is designed to extract and analyze keywords from a dataset of Reddit posts. The script connects to a MongoDB database, retrieves the data, cleans the text, and processes it using various NLP techniques to extract meaningful keywords. The extracted keywords are then scored based on their document frequency and term frequency.

### Dependencies

The script requires the following Python libraries:

- pandas
- numpy
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
- rapidfuzz
- kneed
- tensorflow_hub

### Functions

The script contains the following main functions:

1. `mongo_to_df(filter_day=7)`: Connects to a MongoDB database and retrieves data as a DataFrame.
2. `easy_clean_use(doc)`: Cleans the document for use with the Universal Sentence Encoder (USE) model.
3. `easy_clean_mpnet(text)`: Cleans the text for use with the MPNet model.
4. `clean_text(title, selftext, method='mpnet')`: Cleans the text based on the specified method (either 'mpnet' or 'use').
5. `gb_year_week(col_name, reddit_df=reddit_df)`: Groups the data by year and week.
6. `df_tf_score(kw_li, doc_li)`: Calculates document frequency and term frequency scores for a list of keywords.

### Usage

To use the script, replace the placeholders (e.g., 'your_host', 'your_username', 'your_password', etc.) with your actual values for connecting to the MongoDB database. Also, replace the placeholders in the second code snippet (e.g., 'take_kw', 'y', 'overall_kw_df') with your actual data.

After setting up the required values, run the script to process the data and extract keywords. The output will be a dictionary containing the extracted keywords, their document frequency, and their popularity score.

### Example

```python
import keyword_extraction_pipeline as kep

# Connect to MongoDB and retrieve data as a DataFrame
reddit_df = kep.mongo_to_df(365)

# Clean the text using the MPNet method
reddit_df['doc_mpnet'] = reddit_df.apply(lambda x: kep.clean_text(x.title, x.selftext, method='mpnet'), axis=1)

# Group the data by year and week
mpnet_posts_df = kep.gb_year_week('doc_mpnet')

# Calculate document frequency and term frequency scores for a list of keywords
kw_li = ['example_keyword1', 'example_keyword2']
doc_li = ['example_document1', 'example_document2']
df_score, tf_score = kep.df_tf_score(kw_li, doc_li)

print("Mean document frequency score:", df_score)
print("Mean term frequency score:", tf_score)
```

This example demonstrates how to use the functions in the `keyword_extraction_pipeline2.py` script to connect to a MongoDB database, retrieve data, clean the text, group the data by year and week, and calculate document frequency and term frequency scores for a list of keywords.