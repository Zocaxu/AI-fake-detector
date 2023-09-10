import torch
from PIL import Image
import open_clip
import os
import click
from tqdm import tqdm
import numpy as np

def load_model(model_name, pretrain):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrain)
    return model, preprocess

@click.command()
@click.argument("datadir", type=click.Path(exists=True))
@click.option("--outdir", default="./out")
@click.option("--model", default="ViT-L-14")
@click.option("--pretrain", default="datacomp_xl_s13b_b90k")
@click.option("--n_total", default=200000)
def encode_dataset(datadir: click.Path, outdir: click.Path, model, pretrain, n_total):
    model, preprocess = load_model(model, pretrain)
    count = 0
    if(not os.path.exists(outdir)):
        os.mkdir(outdir)
        os.mkdir(os.path.join(outdir, "fake"))
        os.mkdir(os.path.join(outdir, "real"))
    N = n_total // 40
    for subdir, dirs, files in os.walk(datadir):
        if (str(subdir)[-1] == 'e'):
            label = "fake"
        else:
            label = "real"
        if (len(files) < N):
            print(f"directory {str(subdir)} has size {len(files)} < {N}. Continuing...")
            continue
        random_split = np.random.choice(files, N)
        print(f"Generating {N} embeddings of directory {subdir}...")
        for file in tqdm(random_split):
            image = preprocess(Image.open(os.path.join(subdir,file))).unsqueeze(0)
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = model.encode_image(image)
            torch.save(image_features, os.path.join(outdir, label, str(count)))
            count += 1

if __name__ == "__main__":
    encode_dataset()