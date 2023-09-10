import torch
from PIL import Image
import os
import click
from tqdm import tqdm

@click.command()
@click.argument("base")
@click.argument("out")
@click.option("--n", default=10)
@click.option("--mode", default="number")
def group_step_2(base, out, n, mode):

    classes = [
        "airplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        'diningtable',
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor"
    ]

    fake_features_list = []
    real_features_list = []

    if mode == "number":
        for i in tqdm(range(1, n+1)):
            fake_feature_path = os.path.join(base + f"{i}", "fake_features")
            fake_features = torch.load(fake_feature_path)
            fake_features_list.append(fake_features)

            real_feature_path = os.path.join(base + f"{i}", "real_features")
            real_features = torch.load(real_feature_path)
            real_features_list.append(real_features)
        
        fake_features_cat = torch.cat(fake_features_list, dim=0)
        real_features_cat = torch.cat(real_features_list, dim=0)

        print(fake_features_cat.shape)
        print(real_features_cat.shape)

        torch.save(fake_features_cat, os.path.join(out, "fake_features"))
        torch.save(real_features_cat, os.path.join(out, "real_features"))

    elif mode == "class":
        for c in tqdm(classes):
            fake_feature_path = os.path.join(base + f"{c}", "fake_features")
            fake_features = torch.load(fake_feature_path)
            fake_features_list.append(fake_features)

            real_feature_path = os.path.join(base + f"{c}", "real_features")
            real_features = torch.load(real_feature_path)
            real_features_list.append(real_features)
        
        fake_features_cat = torch.cat(fake_features_list, dim=0)
        real_features_cat = torch.cat(real_features_list, dim=0)

        print(fake_features_cat.shape)
        print(real_features_cat.shape)

        torch.save(fake_features_cat, os.path.join(out, "fake_features"))
        torch.save(real_features_cat, os.path.join(out, "real_features"))

    else:
        print("Mode must be number or class")


if __name__ == "__main__":
    group_step_2()