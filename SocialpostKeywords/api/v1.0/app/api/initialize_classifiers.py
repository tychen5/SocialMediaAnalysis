from transformers import pipeline
import pickle
import os

current_directory = os.path.dirname(__file__)
models_path = os.path.join(current_directory, 'trm_clfs.pkl')


def init_clfs():
    """
    Initialize classifiers using different pre-trained models.
    
    Returns:
        tuple: A tuple containing four zero-shot-classification pipelines.
    """
    classifier = pipeline("zero-shot-classification",
                          model="facebook/bart-large-mnli", num_workers=os.cpu_count())
    classifier2 = pipeline("zero-shot-classification",
                           model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", num_workers=os.cpu_count())
    classifier3 = pipeline("zero-shot-classification",
                           model="joeddav/xlm-roberta-large-xnli", num_workers=os.cpu_count())
    classifier4 = pipeline("zero-shot-classification",
                           model="typeform/distilbert-base-uncased-mnli", num_workers=os.cpu_count())
    return classifier, classifier2, classifier3, classifier4


classifier, classifier2, classifier3, classifier4 = init_clfs()
pickle.dump(obj=(classifier, classifier2, classifier3, classifier4), file=open(models_path, 'wb'))