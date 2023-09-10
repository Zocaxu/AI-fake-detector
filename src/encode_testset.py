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
@click.argument("testdir", type=click.Path(exists=True))
@click.option("--outdir", default="./out")
@click.option("--model", default="ViT-L-14")
@click.option("--pretrain", default="datacomp_xl_s13b_b90k")
@click.option("--feature_size", default=768)
def encode_testset(testdir: click.Path, outdir: click.Path, model, pretrain, feature_size):

    if(not os.path.exists(outdir)):
        os.mkdir(outdir)
    
    model, preprocess = load_model(model, pretrain)
    
    test_dirs = os.listdir(testdir)
    print("===== Generating embedings for directories:")
    print(test_dirs)

    for dir in test_dirs:
        try:
            print(f"Generating embeddings of directory {dir}...")
            files = os.listdir(os.path.join(testdir, dir))
            embeddings = torch.zeros((len(files),feature_size))
            count = 0
            for file in tqdm(files):
                image = preprocess(Image.open(os.path.join(testdir, dir ,file))).unsqueeze(0)
                with torch.no_grad(), torch.cuda.amp.autocast():
                    image_features = model.encode_image(image)
                embeddings[count] = image_features
                count += 1
            
            torch.save(embeddings, os.path.join(outdir, dir + ".pt"))
        except:
            continue


if __name__ == "__main__":
    encode_testset()