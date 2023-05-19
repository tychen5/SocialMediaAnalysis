# Initialize Classifiers

This Python script, `initialize_classifiers.py`, is designed to initialize four different zero-shot-classification pipelines using pre-trained models from the Hugging Face Transformers library. The classifiers are then saved as a pickle file for easy loading and usage in other parts of the project.

### Dependencies

- `transformers`: The Hugging Face Transformers library is used to create the zero-shot-classification pipelines.
- `pickle`: The Python standard library module for serializing and de-serializing Python objects.
- `os`: The Python standard library module for interacting with the operating system.

### Pre-trained Models

The script initializes four classifiers using the following pre-trained models:

1. `facebook/bart-large-mnli`
2. `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
3. `joeddav/xlm-roberta-large-xnli`
4. `typeform/distilbert-base-uncased-mnli`

### Functions

#### `init_clfs()`

This function initializes the classifiers using the pre-trained models mentioned above. It returns a tuple containing the four zero-shot-classification pipelines.

### Usage

Upon execution, the script initializes the classifiers and saves them as a pickle file named `trm_clfs.pkl` in the same directory as the script. This pickle file can then be loaded and used in other parts of the project to perform zero-shot classification tasks.

### Example

```python
import pickle

# Load the classifiers from the pickle file
with open('trm_clfs.pkl', 'rb') as file:
    classifier, classifier2, classifier3, classifier4 = pickle.load(file)

# Use the classifiers for zero-shot classification tasks
labels = ['label1', 'label2', 'label3']
text = "This is a sample text for classification."

result1 = classifier(text, labels)
result2 = classifier2(text, labels)
result3 = classifier3(text, labels)
result4 = classifier4(text, labels)
```