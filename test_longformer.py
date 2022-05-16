import numpy as np
import torch
import torch.nn.functional as F
from transformers import LongformerForSequenceClassification
import dataset_labels
from load_data import collect_transformer_dataset

PATH = './test_trainer/checkpoint-1000'

"""
Load model from specified checkpoint file (see PATH) and test.
NB: inference is slow, can use GPU tensors where possible
"""
def test_transformer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = collect_transformer_dataset()
    test_y = dataset['test']['labels']
    test_y = test_y.to(device)
    test_x = torch.cat((dataset['test']['input_ids'], dataset['test']['attention_mask']), dim=-1)
    test_x = test_x.to(device)

    test_data = torch.utils.data.TensorDataset(test_x, test_y)
    dataloader = torch.utils.data.DataLoader(test_data, batch_size=10, shuffle=False)

    num_labels = len(dataset_labels.idx_to_article)
    print('Loading model')
    model = LongformerForSequenceClassification.from_pretrained(PATH, num_labels=num_labels)
    model = model.to(device)
    print('Model loaded')

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs.logits)
            _, predicted = torch.max(probs, 1)
            print(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print(f'Accuracy so far: {100 * correct // total} %')

    print(f'Accuracy of the model on 1000 test cases: {100 * correct // total} %')

def main():
    test_transformer()

if __name__ == '__main__':
    main()