import numpy as np
import torch
from datasets import load_dataset
from transformers import RobertaTokenizerFast
import labels

MAX_SEQ_LEN = 20000

'''
Convert list of facts into a single string
Convert string labels into integers
'''
def combine_facts(example):
    combined_facts = ''
    for fact in example['facts']:
        combined_facts += fact[4:]
    example['facts'] = combined_facts[:MAX_SEQ_LEN]

    unclear = len(labels.idx_to_article) - 1
    label_list = []
    if not example['labels']:
        label_list.append(unclear)
    else:
        for label in example['labels']:
            if label in labels.label_to_idx:
                label_list.append(labels.label_to_idx[label])
            else:
                label_list.append(unclear)
    example['labels'] = min(label_list)
    return example

def main():
    train_dataset = load_dataset('ecthr_cases', 'alleged-violation-prediction', split='train')
    train_dataset = train_dataset.remove_columns(['silver_rationales', 'gold_rationales'])
    train_dataset = train_dataset.map(combine_facts)

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    encoded_dataset = train_dataset.map(lambda examples: tokenizer(examples['facts'], padding=True), batched=True)
    encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    dataloader = torch.utils.data.DataLoader(encoded_dataset, batch_size=32)
    print(next(iter(dataloader)))

if __name__ == '__main__':
    main()