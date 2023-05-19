# Topic Extraction with BERTopic and KeyBERT

This Python script demonstrates the use of BERTopic and KeyBERT libraries for topic extraction and keyword extraction from a given text document. The script installs the necessary packages, imports required libraries, sets environment variables, and provides an example of how to use the BERTopic and KeyBERT models for topic and keyword extraction.

### Dependencies

- keyphrase-vectorizers
- bertopic
- keybert
- pandas
- numpy
- pickle
- datetime
- pymongo
- itertools
- re
- string
- html
- inflect
- pandarallel
- difflib
- sklearn

### Installation

To install the required packages, run the following command:

```bash
pip install keyphrase-vectorizers bertopic keybert pandas numpy pymongo itertools re string html inflect pandarallel difflib sklearn
```

### Usage

1. Replace the example document `doc` with your desired text document.
2. Set the `worker_num` variable to the number of workers you want to use for parallel processing.
3. Run the script to extract topics and keywords from the given document.

### Key Components

- **BERTopic**: A topic modeling library that leverages BERT embeddings to create topics and extract keywords from text documents.
- **KeyBERT**: A keyword extraction library that uses BERT embeddings to extract keywords from text documents.
- **Fowlkes-Mallows score**: A measure of similarity between two sets of clusterings, used in this script as an example for evaluating the quality of the extracted topics.

### Example

The script provides an example document and demonstrates how to extract topics and keywords using the BERTopic and KeyBERT models. The example document is as follows:

```
"I'm building an office network for about 150 employees. There will be about 100 desktop computers connected by wire. 30-60 other wired devices like video conference devices and cameras. 200-300 wireless devices. Would the gear below be sufficient or am I missing or having too much something? We don't host anything on site and all the users are quite light weight users just sending emails, making video calls, using their phones, instant messaging."
```

After setting the `worker_num` and initializing the KeyphraseCountVectorizer, the script extracts keywords using the KeyBERT model with the following parameters:

- `vectorizer`: The initialized KeyphraseCountVectorizer
- `diversity`: A value of 0.2 to control the diversity of the extracted keywords
- `use_mmr`: Set to True to use Maximal Marginal Relevance for keyword selection
- `top_n`: The number of top keywords to extract (set to 10 in this example)

### Output

The script will output the extracted topics and keywords for the given document.