import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dataset_labels
from models import Classifier
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import LongformerForSequenceClassification, LongformerTokenizer

CLS_PATH = './classifier_model2.pth'
LFORMER_PATH = './test_trainer/checkpoint-1000'
MAX_SEQ_LEN = 4096

"""
Conduct inference on an input string using the baseline Classifier model.
Loads saved TF-IDF vectorizer using pickle and model weights from state dict.
"""
def classifier_inference(text):
    tfidf = pickle.load(open("tfidf.pickle", "rb"))
    x = torch.tensor(tfidf.transform([text]).toarray()).float()
    classes = dataset_labels.idx_to_article

    model = Classifier(input_dim=x.shape[-1], num_classes=len(classes))
    model.load_state_dict(torch.load(CLS_PATH))

    output = model(x)
    _, predicted = torch.max(output.data, 1)
    return predicted

"""
Conduct inference on an input string using the Longformer model.
Uses pretrained tokenizer and fine-tuned model weights, loaded
from LFORMER_PATH.
Tensors will be on GPU where possible.
"""
def longformer_inference(text):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    data = tokenizer(text[-MAX_SEQ_LEN:], padding='max_length', max_length=MAX_SEQ_LEN)
    x = torch.cat((torch.tensor(data['input_ids']),
                   torch.tensor(data['attention_mask'])), dim=0).reshape(1, -1)
    x = x.to(device)

    num_labels = len(dataset_labels.idx_to_article)
    model = LongformerForSequenceClassification.from_pretrained(LFORMER_PATH, num_labels=num_labels)
    model = model.to(device)

    with torch.no_grad():
        output = model(x)
    probs = F.softmax(output.logits, dim=-1)
    _, predicted = torch.max(probs, 1)
    return predicted.item()

if __name__ == '__main__':
    n_args = len(sys.argv)
    if n_args == 1:
        print('Enter a string to classify as an argument')
    else:
        longformer_inference(sys.argv[1])