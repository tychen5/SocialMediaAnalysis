# Keyword Extraction and Language Detection

This Python script demonstrates the use of the KeyBERT library for keyword extraction and the spaCy library for language detection. The script extracts keywords from a given text using two different methods: MMR (Maximal Marginal Relevance) and MaxSum. Additionally, it detects the language of a given text using the spaCy library.

### Dependencies

- keybert
- spacy
- spacy-langdetect

Make sure to install the required packages before running the script.

### Usage

1. Import the required libraries:

```python
from keybert import KeyBERT
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector
```

2. Define a sample document for keyword extraction:

```python
doc = """
         ...
      """
```

3. Initialize the KeyBERT model and extract keywords using MMR and MaxSum methods:

```python
kw_model = KeyBERT()

keywords_mmr = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 3),
                                         use_mmr=True, diversity=0.2, stop_words='english')

keywords_maxsum = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 3),
                                            use_maxsum=True, nr_candidates=20, top_n=5, stop_words=None)
```

4. Define a function to get the language detector:

```python
def get_lang_detector(nlp, name):
    return LanguageDetector()
```

5. Initialize the spaCy model with the language detector:

```python
nlp = spacy.load("xx_ent_wiki_sm")
nlp.add_pipe('sentencizer')
nlp.add_pipe('language_detector', last=True)
```

6. Define a sample text for language detection:

```python
text = '私は知ることを望まない？それで、あなたは知りたいですか？寝たい'
```

7. Process the text with the spaCy model and print the detected language and sentences:

```python
doc = nlp(text)

print(doc._.language)

for sent in doc.sents:
    print(sent.text)
```

### Output

The script will output the extracted keywords using MMR and MaxSum methods, as well as the detected language and sentences of the given text.