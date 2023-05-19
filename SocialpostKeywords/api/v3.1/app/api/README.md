# Keyword Extraction

This Python script, `keyword_extraction.py`, is designed to extract and analyze keywords from a given text input. The script processes the input text and identifies the most relevant keywords, which can be used for various purposes such as search engine optimization, text summarization, or topic modeling.

### Features

- Tokenization: The script breaks down the input text into individual words (tokens) for further processing.
- Stopword removal: Common words that do not carry significant meaning (e.g., "the", "and", "is") are removed from the list of tokens.
- Stemming: Words are reduced to their root form to ensure that different forms of the same word are treated as a single keyword (e.g., "running" and "runner" are both reduced to "run").
- Frequency analysis: The script calculates the frequency of each keyword in the input text.
- Keyword ranking: Keywords are ranked based on their frequency, and the most relevant keywords are returned as output.

### Dependencies

The script relies on the following Python libraries:

- `nltk`: A popular natural language processing library used for tokenization, stopword removal, and stemming.
- `collections`: A built-in Python library used for frequency analysis and keyword ranking.

### Usage

To use the `keyword_extraction.py` script, simply import it into your Python project and call the `extract_keywords` function with the desired input text:

```python
from keyword_extraction import extract_keywords

text = "Your input text goes here."
keywords = extract_keywords(text)

print(keywords)
```

The `extract_keywords` function will return a list of the most relevant keywords found in the input text.

### Customization

You can customize the behavior of the script by modifying the following parameters:

- `num_keywords`: The number of keywords to return as output (default is 10).
- `stopwords`: A list of custom stopwords to remove from the input text (default is the NLTK English stopwords list).
- `stemmer`: The stemming algorithm to use (default is the NLTK PorterStemmer).

### License

This script is released under the MIT License. Feel free to use, modify, and distribute it as needed.