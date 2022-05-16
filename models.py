import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1000)
        self.a1 = nn.ReLU()
        self.fc2 = nn.Linear(1000, 500)
        self.a2 = nn.ReLU()
        self.fc3 = nn.Linear(500, num_classes)
        self.s_max = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.a1(self.fc1(x))
        x = self.a2(self.fc2(x))
        x = self.s_max(self.fc3(x))
        return x
