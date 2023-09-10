import torch
from PIL import Image
import os
import click
from tqdm import tqdm

@click.command()
@click.argument("datadir", type=click.Path(exists=True))
@click.option("--feature_size", default=768)
def group_features(datadir, feature_size):
    list_dir = os.listdir(os.path.join(datadir, 'fake'))
    fake_features = torch.empty(size=(len(list_dir),feature_size))
    for i, file in tqdm(enumerate(list_dir)):
        features = torch.load(os.path.join(datadir, 'fake',file))
        fake_features[i] = features
    torch.save(fake_features, os.path.join(datadir, 'fake_features'))
    print(fake_features.shape)

    list_dir = os.listdir(os.path.join(datadir, 'real'))
    real_features = torch.empty(size=(len(list_dir),feature_size))
    for i, file in tqdm(enumerate(list_dir)):
        features = torch.load(os.path.join(datadir, 'real',file))
        real_features[i] = features
    torch.save(real_features, os.path.join(datadir, 'real_features'))
    print(real_features.shape)

if __name__ == "__main__":
    group_features()