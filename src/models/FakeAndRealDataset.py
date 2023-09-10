import torch
from torch.utils.data import Dataset

class FakeAndRealDataset(Dataset):
    def __init__(self, real_features, fake_features, transform=None, target_transform=None):
        self.features = torch.cat((fake_features,real_features), 0)
        fake_labels = torch.full((fake_features.shape[0],1), 1, dtype=torch.float32)
        real_labels = torch.full((real_features.shape[0],1), 0, dtype=torch.float32)
        self.labels = torch.cat((fake_labels, real_labels), 0)
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feat = self.features[idx]
        label = self.labels[idx]
        if self.transform:
            feat = self.transform(feat)
        return feat, label