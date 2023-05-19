from transformers import pipeline
import pickle
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
models_path = os.path.join(current_directory, 'trm_clfs.pkl')


def init_clfs():
    """
    Initialize classifiers with different models and save them as a tuple.
    Returns:
        tuple: A tuple containing four zero-shot-classification pipelines with different models.
    """
    classifier1 = pipeline("zero-shot-classification",
                           model="facebook/bart-large-mnli", num_workers=os.cpu_count())
    classifier2 = pipeline("zero-shot-classification",
                           model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", num_workers=os.cpu_count())
    classifier3 = pipeline("zero-shot-classification",
                           model="joeddav/xlm-roberta-large-xnli", num_workers=os.cpu_count())
    classifier4 = pipeline("zero-shot-classification",
                           model="typeform/distilbert-base-uncased-mnli", num_workers=os.cpu_count())

    return classifier1, classifier2, classifier3, classifier4


def save_classifiers(classifiers, path):
    """
    Save classifiers to a file using pickle.
    Args:
        classifiers (tuple): A tuple containing four zero-shot-classification pipelines.
        path (str): The path to save the classifiers.
    """
    with open(path, 'wb') as file:
        pickle.dump(obj=classifiers, file=file)


if __name__ == "__main__":
    classifiers = init_clfs()
    save_classifiers(classifiers, models_path)
