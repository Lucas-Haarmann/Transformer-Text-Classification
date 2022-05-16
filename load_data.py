import numpy as np
import pickle
import torch
from datasets import load_dataset
from transformers import LongformerTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import dataset_labels

MAX_SEQ_LEN = 4096
NUM_LABELS = len(dataset_labels.idx_to_article)

'''
Takes in, preprocesses and returns huggingface dataset.
Converts list of case facts into a single string.
Strings longer than MAX_SEQ_LEN are truncated at the front
(empirically better results than truncating at the rear).
Converts string labels into integers.
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
    example['labels'] = label_list[0]
    return example

'''
Load ECtHR dataset, preprocess and tokenize using pretrained
Longformer tokenizer.
Returns huggingface dataset.
'''
def collect_transformer_dataset():
    dataset = load_dataset('ecthr_cases', 'alleged-violation-prediction')
    dataset = dataset.remove_columns(['silver_rationales', 'gold_rationales'])
    dataset = dataset.map(combine_facts)

    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    dataset = dataset.map(lambda examples: tokenizer(examples['facts'], padding='max_length', max_length=MAX_SEQ_LEN))
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    dataset = dataset.remove_columns('facts')

    return dataset

"""
Similar to combine_facts, but for the baseline classifier.
There is no MAX_SEQ_LEN restriction.
"""
def combine_facts_classifier(example):
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
    example['labels'] = label_list[0]
    return example

'''
Load, preprocess and vectorize ECtHR dataset from huggingface.
NB 1: TF-IDF hyperparameters have a huge affect on model training and results.
Room for experimentation and improvement here.
NB 2: TF-IDF vectorizer object is saved using pickle. This is necessary
to load the same vectorizer (fitted to train set) for testing and inference.
Returns 3 PyTorch datasets.
'''
def collect_classifier_dataset():
    dataset = load_dataset('ecthr_cases', 'alleged-violation-prediction')
    dataset = dataset.remove_columns(['silver_rationales', 'gold_rationales'])
    dataset = dataset.map(combine_facts_classifier)

    tfidf = TfidfVectorizer(input='content',
                            sublinear_tf=True,
                            strip_accents='ascii',
                            analyzer='word',
                            ngram_range=(1, 1),
                            stop_words='english',
                            max_df=0.9,
                            max_features=2000)
    tfidf = tfidf.fit(dataset['train']['facts'])
    X_train = torch.tensor(tfidf.transform(dataset['train']['facts']).toarray()).float()
    X_val = torch.tensor(tfidf.transform(dataset['validation']['facts']).toarray()).float()
    X_test = torch.tensor(tfidf.transform(dataset['test']['facts']).toarray()).float()

    # Save tf-idf vocab object
    pickle.dump(tfidf, open("tfidf.pickle", "wb"))

    y_train = torch.tensor(dataset['train']['labels'])
    y_val = torch.tensor(dataset['validation']['labels'])
    y_test = torch.tensor(dataset['test']['labels'])

    train_set = torch.utils.data.TensorDataset(X_train, y_train)
    val_set = torch.utils.data.TensorDataset(X_val, y_val)
    test_set = torch.utils.data.TensorDataset(X_test, y_test)

    return train_set, val_set, test_set

if __name__ == '__main__':
    collect_classifier_dataset()