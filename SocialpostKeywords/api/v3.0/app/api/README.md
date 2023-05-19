# Keyword Extraction

This Python script, `keyword_extraction.py`, is designed to extract and analyze keywords from a given text input. The script processes the input text and identifies the most relevant keywords, which can be used for various purposes such as search engine optimization, text summarization, or topic modeling.

### Features

- Tokenization: The script breaks down the input text into individual words (tokens) for further processing.
- Stopword removal: Common words that do not carry significant meaning (e.g., "the", "and", "is") are removed from the list of tokens.
- Stemming: The script reduces words to their root form to ensure that different forms of the same word are treated as a single keyword (e.g., "running" and "runner" are both reduced to "run").
- Frequency analysis: The script calculates the frequency of each keyword in the input text.
- Keyword ranking: The script ranks the keywords based on their frequency and relevance to the input text.

### Dependencies

To run the `keyword_extraction.py` script, you will need the following Python libraries:

- NLTK (Natural Language Toolkit): A popular library for natural language processing tasks, including tokenization, stemming, and stopword removal.
- NumPy: A library for numerical computing in Python, used for efficient frequency analysis.

### Usage

To use the `keyword_extraction.py` script, simply import it into your Python project and call the `extract_keywords` function with the input text as an argument:

```python
from keyword_extraction import extract_keywords

input_text = "Your input text goes here."
keywords = extract_keywords(input_text)

print("Keywords:", keywords)
```

The `extract_keywords` function will return a list of keywords ranked by their relevance to the input text.

### Customization

You can customize the behavior of the `keyword_extraction.py` script by modifying the following parameters:

- `stopwords`: A list of words to be excluded from the keyword extraction process. By default, this list includes common English stopwords, but you can add or remove words as needed.
- `stemmer`: The stemming algorithm used to reduce words to their root form. By default, the script uses the Porter Stemmer from the NLTK library, but you can replace it with any other stemming algorithm supported by NLTK.

### License

This script is released under the MIT License. Feel free to use, modify, and distribute it as needed.