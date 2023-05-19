
# Zero-Shot Classifiers Setup

This Python script, `zero_shot_classifiers_setup.py`, initializes and saves four different zero-shot classifiers using the Hugging Face Transformers library. These classifiers can be used for various natural language understanding tasks, such as text classification, sentiment analysis, and more.

## Dependencies

- Python 3.6 or higher
- Hugging Face Transformers library

## Usage

1. Set the `current_directory` variable to the path where you want to save the classifiers.
2. Run the script using `python zero_shot_classifiers_setup.py`.

## Classifiers

The script initializes the following zero-shot classifiers:

1. `facebook/bart-large-mnli`: A BART model fine-tuned on the Multi-Genre Natural Language Inference (MNLI) dataset.
2. `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`: A DeBERTa model fine-tuned on a combination of MNLI, FEVER, ANLI, and Ling/Wanli datasets.
3. `joeddav/xlm-roberta-large-xnli`: An XLM-RoBERTa model fine-tuned on the Cross-lingual Natural Language Inference (XNLI) dataset.
4. `typeform/distilbert-base-uncased-mnli`: A DistilBERT model fine-tuned on the MNLI dataset.

## Functions

- `init_clfs()`: Initializes the classifiers and returns them as a tuple.

## Output

The script saves the initialized classifiers as a pickle file named `trm_clfs.pkl` in the specified directory.