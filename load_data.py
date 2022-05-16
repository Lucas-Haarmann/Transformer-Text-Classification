import numpy as np
import torch
from datasets import load_dataset
from transformers import LongformerTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import dataset_labels

MAX_SEQ_LEN = 4096
NUM_LABELS = len(dataset_labels.idx_to_article)

'''
Convert list of facts into a single string
Convert string labels into integers
'''
def combine_facts(example):
    combined_facts = ''
    for fact in example['facts']:
        combined_facts += fact[4:]
    example['facts'] = combined_facts[-MAX_SEQ_LEN:]

    unclear = len(dataset_labels.idx_to_article) - 1
    label_list = []
    if not example['labels']:
        label_list.append(unclear)
    else:
        for label in example['labels']:
            if label in dataset_labels.label_to_idx:
                label_list.append(dataset_labels.label_to_idx[label])
            else:
                label_list.append(unclear)
    example['labels'] = min(label_list)
    return example

def collect_transformer_dataset():
    dataset = load_dataset('ecthr_cases', 'alleged-violation-prediction')
    dataset = dataset.remove_columns(['silver_rationales', 'gold_rationales'])
    dataset = dataset.map(combine_facts)

    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    dataset = dataset.map(lambda examples: tokenizer(examples['facts'], padding='max_length', max_length=MAX_SEQ_LEN))
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    dataset = dataset.remove_columns('facts')

    return dataset

def preprocess_classifier(example, vectorizer):
    combined_facts = ''
    for fact in example['facts']:
        combined_facts += fact[4:]
    example['facts'] = combined_facts

    unclear = len(dataset_labels.idx_to_article) - 1
    label_list = []
    if not example['labels']:
        label_list.append(unclear)
    else:
        for label in example['labels']:
            if label in dataset_labels.label_to_idx:
                label_list.append(dataset_labels.label_to_idx[label])
            else:
                label_list.append(unclear)
    example['labels'] = min(label_list)
    return example

def collect_classifier_dataset():
    dataset = load_dataset('ecthr_cases', 'alleged-violation-prediction')
    dataset = dataset.remove_columns(['silver_rationales', 'gold_rationales'])
    dataset = dataset.map(combine_facts)

    tfidf = TfidfVectorizer(input='content',
                            sublinear_tf=True,
                            analyzer='word',
                            ngram_range=(1, 1),
                            stop_words='english')
    X_train = torch.tensor(tfidf.fit_transform(dataset['train']['facts']).toarray()).float()
    X_val = torch.tensor(tfidf.transform(dataset['validation']['facts']).toarray()).float()
    X_test = torch.tensor(tfidf.transform(dataset['test']['facts']).toarray()).float()

    y_train = torch.tensor(dataset['train']['labels'])
    y_val = torch.tensor(dataset['validation']['labels'])
    y_test = torch.tensor(dataset['test']['labels'])

    train_set = torch.utils.data.TensorDataset(X_train, y_train)
    val_set = torch.utils.data.TensorDataset(X_val, y_val)
    test_set = torch.utils.data.TensorDataset(X_test, y_test)

    return train_set, val_set, test_set