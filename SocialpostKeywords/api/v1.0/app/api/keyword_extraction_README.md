# Keyword Extraction

This Python script, `keyword_extraction.py`, is designed to extract relevant keywords from a given text input. The script utilizes natural language processing techniques to identify and return the most important words or phrases that best represent the main topics discussed in the input text.

### Features

- Tokenization: The script breaks down the input text into individual words or tokens.
- Stopword removal: Common words that do not carry significant meaning are filtered out.
- Lemmatization: Words are reduced to their base or dictionary form, which helps in identifying similar words with different forms.
- Term frequency-inverse document frequency (TF-IDF): This statistical measure is used to evaluate the importance of each word in the input text.
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

text = "This is a sample text to demonstrate keyword extraction."
keywords = extract_keywords(text)

print(keywords)
```

### Customization

You can customize the number of keywords returned by the script by modifying the `num_keywords` parameter in the `extract_keywords` function. By default, it is set to 10.

```python
keywords = extract_keywords(text, num_keywords=5)
```

### Conclusion

The `keyword_extraction.py` script is a powerful tool for extracting meaningful keywords from any given text input. It can be easily integrated into various applications, such as search engines, content summarization, and topic modeling.