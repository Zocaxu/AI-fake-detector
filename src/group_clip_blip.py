import torch
import os
import numpy as np
from tqdm import tqdm
import click


@click.command()
@click.argument("datadir", type=click.Path(exists=True))
@click.option("--caption_mode", default="blip")
@click.option("--feature_size", default=1536)
def group_blip(datadir, caption_mode, feature_size):
    
    if caption_mode not in ["blip", "natural"]:
        print("caption_mode has to be blip or natural")
        exit(1)
    
    fake_features_path = os.path.join(datadir, "fake", caption_mode, "features")
    real_features_path = os.path.join(datadir, "real", caption_mode, "features")

    fake_features_list = np.array(os.listdir(fake_features_path))
    real_features_list = np.array(os.listdir(real_features_path))

    fake_features = torch.empty(size=(len(fake_features_list),feature_size))
    for i, file in tqdm(enumerate(fake_features_list)):
        features = torch.load(os.path.join(fake_features_path, file))
        fake_features[i] = features
    torch.save(fake_features, os.path.join(datadir, "fake_features"))
    print(fake_features.shape)

    real_features = torch.empty(size=(len(real_features_list),feature_size))
    for i, file in tqdm(enumerate(real_features_list)):
        features = torch.load(os.path.join(real_features_path, file))
        real_features[i] = features
    torch.save(real_features, os.path.join(datadir, "real_features"))
    print(real_features.shape)










if __name__ == "__main__":
    group_blip()