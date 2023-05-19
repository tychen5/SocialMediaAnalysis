# Reddit Keyword Analysis

This Python script is designed to analyze Reddit posts and classify them based on a list of predefined keywords. The script uses multiple zero-shot classification models from the Hugging Face Transformers library to perform the classification. It also connects to a MongoDB database to retrieve Reddit post data and processes the text data for analysis.

### Dependencies

The script requires the following libraries:

- pandas
- numpy
- transformers
- tqdm
- pymongo
- datetime
- html
- re
- pickle
- functools

### Functionality

The script performs the following tasks:

1. Imports a list of keywords from an Excel file and combines them with their similar words.
2. Connects to a MongoDB database and retrieves Reddit post data.
3. Cleans the text data by removing emails, URLs, emojis, and other unwanted characters.
4. Splits the text data into paragraphs for further analysis.
5. Loads a checkpoint file containing previously analyzed keywords and their associated timestamps.
6. Classifies the Reddit posts using multiple zero-shot classification models from the Hugging Face Transformers library.

### Usage

Before running the script, make sure to replace the placeholders with your actual file paths and MongoDB credentials.

To run the script, simply execute the following command:

```
python reddit_keyword_analysis.py
```

The script will output the classified Reddit posts based on the predefined keywords.