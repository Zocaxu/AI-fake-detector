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
    
    model, preprocess = load_model(model, pretrain)

    for root, dirs, files in os.walk(testdir):
        if len(files) == 0:
            continue
        rootname, dirname = os.path.split(root)
        if dirname == '0_real' or dirname == '1_fake':
            testset = os.path.basename(rootname)
            out_path = os.path.join(outdir, testset, dirname)
        else:
            model_path, submodel = os.path.split(dirname)
            testset = os.path.basename(model_path)
            out_path = os.path.join(outdir, testset, submodel, dirname)
        os.makedirs(out_path, exist_ok = True)
        
        print(f"Generating embeddings of directory {root}...")
        embeddings = torch.zeros((len(files),feature_size))
        count = 0
        for file in tqdm(files):
            image = preprocess(Image.open(os.path.join(root,file))).unsqueeze(0)
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = model.encode_image(image)
            embeddings[count] = image_features
            count += 1
            
        torch.save(embeddings, os.path.join(out_path, testset + ".pt"))


if __name__ == "__main__":
    encode_testset()