import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from load_data import collect_classifier_dataset
import dataset_labels
from models import Classifier

PATH = './classifier_model2.pth'

def test_classifier():
    train_set, val_set, test_set = collect_classifier_dataset()
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False, num_workers=2)
    classes = dataset_labels.idx_to_article

    example = iter(test_loader).next()[0]
    input_dim = example.shape[-1]
    model = Classifier(input_dim=input_dim, num_classes=len(classes))
    model.load_state_dict(torch.load(PATH))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            print(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on 1000 test cases: {100 * correct // total} %')


if __name__ == "__main__":
    test_classifier()