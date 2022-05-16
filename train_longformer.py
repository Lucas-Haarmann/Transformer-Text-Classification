import numpy as np
import torch
from datasets import load_metric
from transformers import LongformerForSequenceClassification, TrainingArguments, Trainer
import dataset_labels
from load_data import collect_transformer_dataset

def eval_func(metric, output):
    logits, labels = output
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)

def train_transformer():
    dataset = collect_transformer_dataset()
    model = LongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096", num_labels=NUM_LABELS)
    metric = load_metric('accuracy')
    train_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
    trainer = Trainer(model=model,
                      args=train_args,
                      train_dataset=dataset['train'],
                      eval_dataset=dataset['validation'],
                      compute_metrics=eval_func)
    trainer.train()

def main():
    train_transformer()

if __name__ == '__main__':
    main()