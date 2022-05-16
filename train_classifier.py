import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from load_data import collect_classifier_dataset
import dataset_labels
from models import Classifier

PATH = './classifier_model.pth'

def train_classifier():
    train_set, val_set, test_set = collect_classifier_dataset()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=4, shuffle=False, num_workers=2)
    classes = dataset_labels.idx_to_article

    example = iter(train_loader).next()[0]
    input_dim = example.shape[-1]
    print(input_dim)
    model = Classifier(input_dim=input_dim, num_classes=len(classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(3):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    print('Finished Training')
    torch.save(model.state_dict(), PATH)

def main():
    train_classifier()

if __name__ == '__main__':
    main()