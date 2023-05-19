# Keyword Extraction

This Python script, `keyword_extraction.py`, is designed to extract relevant keywords from a given text input. The script utilizes natural language processing techniques to identify and return the most important keywords, which can be used for various purposes such as search engine optimization, text summarization, and document classification.

### Features

- Tokenization: The script breaks down the input text into individual words or tokens.
- Stopword removal: Common words that do not carry significant meaning are filtered out.
- Lemmatization: Words are reduced to their base or dictionary form, which helps in identifying similar words with different forms.
- Term frequency-inverse document frequency (TF-IDF): This statistical measure is used to evaluate the importance of a word in the context of the input text.
- Keyword extraction: The script returns a list of the most relevant keywords based on their TF-IDF scores.

### Dependencies

To run this script, you will need the following Python libraries:

- `nltk`: A popular natural language processing library.
- `sklearn`: A machine learning library that provides the TF-IDF implementation.

### Usage

To use the `keyword_extraction.py` script, simply import it into your Python project and call the `extract_keywords` function with the input text as an argument. The function will return a list of the most relevant keywords.

Example:

```python
from keyword_extraction import extract_keywords

text = "This is a sample text for keyword extraction."
keywords = extract_keywords(text)

print(keywords)
```

### Customization

You can customize the script to better suit your needs by modifying the following parameters:

- `max_keywords`: The maximum number of keywords to return.
- `min_word_length`: The minimum length of a word to be considered as a keyword.
- `stopwords`: A custom list of stopwords to be removed from the input text.

### License

This script is released under the MIT License. Feel free to use, modify, and distribute it as needed.