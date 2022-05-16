import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from load_data import collect_classifier_dataset
import dataset_labels
from models import Classifier
from sklearn.feature_extraction.text import TfidfVectorizer

PATH = './classifier_model2.pth'

def classifier_inference(text):
    tfidf = pickle.load(open("tfidf.pickle", "rb"))
    x = torch.tensor(tfidf.transform([text]).toarray()).float()
    classes = dataset_labels.idx_to_article
    model = Classifier(input_dim=x.shape[-1], num_classes=len(classes))
    model.load_state_dict(torch.load(PATH))
    output = model(x)
    _, predicted = torch.max(output.data, 1)
    return predicted

if __name__ == '__main__':
    n_args = len(sys.argv)
    if n_args == 1:
        print('Enter a string to classify as an argument')
    else:
        classifier_inference(sys.argv[1])