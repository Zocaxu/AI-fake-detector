import torch
import numpy as np
import click
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt

from FakeOrRealClassifier1 import FakeOrRealClassifier1
from FakeOrRealClassifier2 import FakeOrRealClassifier2
from FakeOrRealClassifier3 import FakeOrRealClassifier3

from FakeAndRealDataset import FakeAndRealDataset

def config_loss_function(pos_weight):
    return torch.nn.BCEWithLogitsLoss(reduction='sum', pos_weight=pos_weight)

def config_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=0.0001)

def train_loop(dataloader, model, loss_fn, optim, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss = 0
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss

        loss.backward()
        optim.step()
        optim.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch +1)*len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    train_loss /= num_batches
    return train_loss

def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            sigmoid = torch.nn.Sigmoid()
            pred_sig = sigmoid(model(X)).round()
            correct += (pred_sig == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return  test_loss

@click.command()
@click.argument("fake_set", type=click.Path(exists=True))
@click.argument("real_set", type=click.Path(exists=True))
@click.option("--out", default=None)
@click.option("--epochs", default=10)
@click.option("--weight", default=1.0)
@click.option("--ablation", default=None)
def train_model(fake_set, real_set, out, epochs, weight, ablation):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    print("===== Loading dataset...")
    fake_features = torch.load(fake_set)
    print(f"fake set size: {fake_features.shape}")
    real_features = torch.load(real_set)
    print(f"real set size: {real_features.shape}")

    if ablation is not None:
        n_fakes = len(fake_features)
        n_abl = int(n_fakes * int(ablation) / 100)
        indexes = torch.randperm(n_fakes)[:n_abl]
        fake_features = fake_features[indexes]
        real_features = real_features[indexes]
        print("Sets ablated to ", n_abl, " each")

    fake_train, fake_test = train_test_split(fake_features, test_size=0.2)
    real_train, real_test = train_test_split(real_features, test_size=0.2)

    train_dataset = FakeAndRealDataset(real_train, fake_train)
    test_dataset = FakeAndRealDataset(real_test, fake_test)
    print(f"training set size: {len(train_dataset)}")
    print(f"test set size: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=168, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=168, shuffle=True)

    _, input_dimension = fake_features.shape
    model = FakeOrRealClassifier1(input_dimension)
    model = model.to(device)

    """Train the model"""
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)

    positive_weight = weight

    criterion = config_loss_function(torch.tensor([positive_weight]).to(device))
    optimizer = config_optimizer(model)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train_loop(train_loader, model, criterion, optimizer, device)
        test_loss = test_loop(test_loader, model, criterion, device)
        train_losses[t] = train_loss
        test_losses[t] = test_loss
    
    X = range(epochs)
    plt.plot(X, train_losses, label='train loss')
    plt.plot(X, test_losses, label='test loss')
    plt.ylim(0)
    plt.legend()

    if out is not None:
        torch.save(model.state_dict(), out)
        plt.savefig(out + ".png")
        print(f"Model saved to location {out}")


if __name__ == "__main__":
    train_model()