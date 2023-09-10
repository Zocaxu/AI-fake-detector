import torch

class FakeOrRealClassifier3(torch.nn.Module):
    def __init__(self, input_dimension):
        super().__init__()
        self.l1 = torch.nn.Linear(input_dimension, 2*input_dimension)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.l2 = torch.nn.Linear(2*input_dimension, 3*input_dimension)
        self.l3 = torch.nn.Linear(3*input_dimension, 1)
    def forward(self, X):
        l1_out = self.l1(X)
        dropout = self.dropout(l1_out)
        l2_out = self.l2(dropout)
        l3_out = self.l3(l2_out)
        return l3_out