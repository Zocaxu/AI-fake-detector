import torch

class FakeOrRealClassifier2(torch.nn.Module):
    def __init__(self, input_dimension):
        super().__init__()
        self.l1 = torch.nn.Linear(input_dimension, 2*input_dimension)
        self.l2 = torch.nn.Linear(2*input_dimension, 1)
    def forward(self, X):
        l1_out = self.l1(X)
        l2_out = self.l2(l1_out)
        return l2_out