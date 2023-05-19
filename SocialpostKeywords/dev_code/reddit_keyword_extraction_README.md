`reddit_keyword_extraction.py`

README.md:

# Reddit Keyword Extraction

This Python script, `reddit_keyword_extraction.py`, is designed to extract keywords from Reddit posts by connecting to a MongoDB database, cleaning and processing the text data, and then using the KeyBERT library to extract relevant keywords.

## Dependencies

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

## Functionality

The script performs the following tasks:

1. Connects to a MongoDB database and retrieves Reddit posts data.
2. Cleans and processes the text data from the Reddit posts.
3. Groups the data by year and week.
4. Uses the KeyBERT library to extract keywords from the processed text data.

## Functions

- `mongo_to_df(filter_day=7)`: Connects to a MongoDB database and retrieves Reddit posts data, returning a DataFrame.
- `easy_clean_use(doc)`: Cleans and processes the text data for the Universal Sentence Encoder (USE) model.
- `easy_clean_mpnet(text)`: Cleans and processes the text data for the MPNet model.
- `clean_text(title, selftext, method='mpnet')`: Cleans and processes the text data based on the specified method (either 'mpnet' or 'use').
- `gb_year_week(col_name, reddit_df=reddit_df)`: Groups the data by year and week, returning a DataFrame.
- `mpnet_kw(liostr)`: Extracts keywords from the processed text data using the KeyBERT library.

## Usage

1. Replace the placeholders (e.g., 'your_host', 'your_username', 'your_password', etc.) in the script with your actual sensitive information.
2. Run the script to extract keywords from Reddit posts.

## Output

The script generates a DataFrame containing the extracted keywords for each Reddit post. The DataFrame is grouped by year and week, and includes the processed text data and the candidate keywords.

## Note

This script uses the KeyBERT library, which requires a pre-trained model. In this case, the "all-mpnet-base-v2" model is used. Make sure to install the required dependencies and have the necessary model files available before running the script.