import torch
from PIL import Image
import open_clip
import os
import csv
import click
from tqdm import tqdm
import numpy as np

def load_model(model_name, pretrain):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrain)
    return model, preprocess

@click.command()
@click.argument("diffusion_dir", type=click.Path(exists=True))
@click.argument("coco_dir", type=click.Path(exists=True))
@click.argument("coco_csv", type=click.Path(exists=True))
@click.option("--outdir", default="./out")
@click.option("--model", default="ViT-L-14")
@click.option("--pretrain", default="datacomp_xl_s13b_b90k")
def encode_dataset(diffusion_dir: click.Path, coco_dir, coco_csv, outdir: click.Path, model, pretrain):
    model, preprocess = load_model(model, pretrain)

    if(not os.path.exists(outdir)):
        os.mkdir(outdir)
        os.mkdir(os.path.join(outdir, "fake"))
        os.mkdir(os.path.join(outdir, "real"))

    with open(coco_csv, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        count = 0
        for row in tqdm(reader):

            coco_file_name = row[0]

            # Coco image
            n = int(coco_file_name.split('.')[0])
            q = n // 10000
            folder = "{:05d}".format(q)
            coco_image = Image.open(os.path.join(coco_dir, folder, coco_file_name))
            image = preprocess(coco_image).unsqueeze(0)
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = model.encode_image(image)
            torch.save(image_features, os.path.join(outdir, "real", str(count)))

            # Stable diffusion image
            stable_file_name = "{:05d}".format(count) + ".png"
            stable_image = Image.open(os.path.join(diffusion_dir, stable_file_name))
            image = preprocess(stable_image).unsqueeze(0)
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = model.encode_image(image)
            torch.save(image_features, os.path.join(outdir, "fake", str(count)))

            count += 1
    

if __name__ == "__main__":
    encode_dataset()