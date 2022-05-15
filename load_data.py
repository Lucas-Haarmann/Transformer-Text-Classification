import numpy as np
import torch
from datasets import load_dataset, load_metric
from transformers import LongformerTokenizer, LongformerForSequenceClassification, TrainingArguments, Trainer
import labels

MAX_SEQ_LEN = 4096
NUM_LABELS = len(labels.idx_to_article)

'''
Convert list of facts into a single string
Convert string labels into integers
'''
def combine_facts(example):
    combined_facts = ''
    for fact in example['facts']:
        combined_facts += fact[4:]
    example['facts'] = combined_facts[-MAX_SEQ_LEN:]

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
    dataset = load_dataset('ecthr_cases', 'alleged-violation-prediction')
    dataset = dataset.remove_columns(['silver_rationales', 'gold_rationales'])
    dataset = dataset.map(combine_facts)

    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    dataset = dataset.map(lambda examples: tokenizer(examples['facts'], padding='max_length', max_length=MAX_SEQ_LEN))
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    dataset = dataset.remove_columns('facts')

    model = LongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096", num_labels=NUM_LABELS)
    metric = load_metric('accuracy')
    def eval_func(metric, output):
        logits, labels = output
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)
    train_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
    trainer = Trainer(model=model,
                      args=train_args,
                      train_dataset=dataset['train'],
                      eval_dataset=dataset['validation'],
                      compute_metrics=eval_func)
    trainer.train()

if __name__ == '__main__':
    main()