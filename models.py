import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.a1 = nn.ReLU()
        self.fc2 = nn.Linear(100, num_classes)
        self.s_max = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.a1(self.fc1(x))
        x = self.s_max(self.fc2(x))
        return x