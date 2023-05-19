# Initialize and Save Classifiers

This Python script, `initialize_and_save_classifiers.py`, initializes four different zero-shot-classification models using the Hugging Face Transformers library and saves them as a tuple in a pickle file for later use.

### Models Used

The script initializes the following models:

1. `facebook/bart-large-mnli`
2. `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
3. `joeddav/xlm-roberta-large-xnli`
4. `typeform/distilbert-base-uncased-mnli`

These models are chosen for their performance in zero-shot classification tasks.

### Functions

The script contains two main functions:

#### `init_clfs()`

This function initializes the classifiers with different models and returns them as a tuple. It takes no arguments and returns a tuple containing four zero-shot-classification pipelines with different models.

#### `save_classifiers(classifiers, path)`

This function saves the classifiers to a file using the pickle module. It takes two arguments:

- `classifiers (tuple)`: A tuple containing four zero-shot-classification pipelines.
- `path (str)`: The path to save the classifiers.

### Execution

When the script is executed, it initializes the classifiers using the `init_clfs()` function and saves them to a file using the `save_classifiers()` function. The classifiers are saved in a file named `trm_clfs.pkl` in the same directory as the script.

### Dependencies

To run this script, you need to have the following Python packages installed:

- `transformers`
- `pickle`
- `os`

You can install the `transformers` package using pip:

```
pip install transformers
```