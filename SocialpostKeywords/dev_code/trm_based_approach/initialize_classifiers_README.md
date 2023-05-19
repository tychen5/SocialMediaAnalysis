# Initialize Classifiers

This Python script, `initialize_classifiers.py`, is designed to initialize and save four different zero-shot classification models from the Hugging Face Transformers library. These classifiers are based on pre-trained models and can be used for various natural language processing tasks, such as text classification, sentiment analysis, and more.

## Dependencies

- Python 3.6 or higher
- transformers (Hugging Face library)
- pickle

## Models

The script initializes the following four classifiers:

1. `facebook/bart-large-mnli`
2. `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
3. `joeddav/xlm-roberta-large-xnli`
4. `typeform/distilbert-base-uncased-mnli`

These classifiers are based on different pre-trained models and can be used for zero-shot classification tasks.

## Usage

1. Set the `current_directory` variable to the path where you want to save the pickled classifiers.
2. Run the script using `python initialize_classifiers.py`.

The script will initialize the classifiers and save them as a tuple in a pickle file named `trm_clfs.pkl` in the specified directory.

## Functions

- `init_clfs()`: Initializes the classifiers and returns them as a tuple.
- `main()`: Initializes the classifiers using `init_clfs()` and saves them to a pickle file.

## License

This project is licensed under the MIT License.