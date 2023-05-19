# Weekly Keyword Extraction

This Python script is designed to extract weekly keywords from a MongoDB database containing Reddit posts. The script utilizes the KeyBERT library for keyword extraction and the Universal Sentence Encoder (USE) for text embeddings. The extracted keywords can be used for various purposes, such as content analysis, trend identification, and topic modeling.

### Dependencies

The script requires the following libraries:

- os
- time
- random
- gc
- itertools
- re
- string
- html
- inflect
- operator
- datetime
- pymongo
- pandas
- numpy
- pickle
- pandarallel
- sklearn
- keybert
- keyphrase_vectorizers
- rapidfuzz
- kneed
- difflib
- tqdm
- tensorflow_hub

### Functions

The script contains the following functions:

- `mongo_to_df(filter_day=7)`: Connects to a MongoDB database, retrieves Reddit posts created within the specified number of days, and returns a DataFrame containing the relevant data.
- `easy_clean_use(doc)`: Cleans the input text by removing unwanted characters, links, and email addresses.
- `clean_text(title, selftext, method='mpnet')`: Combines the title and selftext of a Reddit post and cleans the text using the specified method.
- `gb_year_week(col_name, reddit_df=reddit_df)`: Groups the DataFrame by year and week, and returns a new DataFrame with aggregated data.
- `use_kw(lioliostr)`: Extracts keywords from the input text using the KeyBERT model and the Universal Sentence Encoder.

### Workflow

1. Retrieve Reddit posts from the MongoDB database using the `mongo_to_df` function.
2. Clean the text of the Reddit posts using the `clean_text` function.
3. Group the cleaned text by year and week using the `gb_year_week` function.
4. Extract keywords from the grouped text using the `use_kw` function.
5. Save the extracted keywords to a pickle file.

### Usage

To use the script, simply run `python weekly_keyword_extraction2.py`. The script will connect to the MongoDB database, retrieve the Reddit posts, clean the text, extract the keywords, and save the results to a pickle file.

Make sure to update the MongoDB connection details, such as host, username, password, port, authSource, and authMechanism, as well as the path to save the pickle file.

### Output

The script outputs a pickle file containing a DataFrame with the extracted keywords for each week. The DataFrame includes the following columns:

- year
- week
- doc_use (cleaned text)
- count (number of posts)
- candidate_keywords (list of extracted keywords)