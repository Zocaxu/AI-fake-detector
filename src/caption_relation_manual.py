from PIL import Image
import requests
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

import os
from tqdm import tqdm
import click
import open_clip
import numpy as np
import csv
import pandas as pd

def load_model(model_name, pretrain):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrain)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer

@click.command()
@click.argument("diffusion_dir", type=click.Path(exists=True))
@click.argument("blip_dir", type=click.Path(exists=True))
@click.argument("coco_dir", type=click.Path(exists=True))
@click.argument("coco_csv", type=click.Path(exists=True))
@click.option("--outdir", default="./out")
@click.option("--clip_model", default="ViT-L-14")
@click.option("--clip_pretrain", default="datacomp_xl_s13b_b90k")
def encode_clip_blip(diffusion_dir, blip_dir, coco_dir, coco_csv, outdir, clip_model, clip_pretrain):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_model, clip_preprocess, clip_tokenizer = load_model(clip_model, clip_pretrain)

    if(not os.path.exists(outdir)):
        os.makedirs(outdir)

    tab = ({
        'Real-BLIP' :[0],
        'Fake-BLIP': [0]
        })
    tab_metrics = pd.DataFrame(tab)

    with open(coco_csv, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        count = 0
        for row in tqdm(reader):

            coco_file_name = row[0]
            coco_caption = row[1]

            # Coco image
            n = int(coco_file_name.split('.')[0])
            q = n // 10000
            folder = "{:05d}".format(q)
            img_path = os.path.join(coco_dir, folder, coco_file_name)
            coco_image = Image.open(img_path)
            clip_image = clip_preprocess(coco_image).unsqueeze(0)
            
            real_blip_path = os.path.join(blip_dir, "real", "blip", "captions", coco_file_name + '.txt')
            with open(real_blip_path, 'r') as cap_file:
                blip_caption = cap_file.readline()

            with torch.no_grad(), torch.cuda.amp.autocast():
                clip_features = clip_model.encode_image(clip_image)
                clip_features = torch.reshape(clip_features, (-1,))
                
                blip_tokenized_caption = clip_tokenizer(blip_caption)
                blip_text_features = clip_model.encode_text(blip_tokenized_caption)
                blip_text_features = torch.reshape(blip_text_features, (-1,))
            
            r_sim = torch.sigmoid(F.cosine_similarity(clip_features, blip_text_features, dim=0)).item()
            
            # Stable diffusion image
            stable_file_name = "{:05d}".format(count) + ".png"
            img_path = os.path.join(diffusion_dir, stable_file_name)
            stable_image = Image.open(os.path.join(diffusion_dir, stable_file_name))
            clip_image = clip_preprocess(stable_image).unsqueeze(0)
            
            fake_blip_path = os.path.join(blip_dir, "fake", "blip", "captions", stable_file_name + '.txt')
            with open(real_blip_path, 'r') as cap_file:
                blip_caption = cap_file.readline()
            
            with torch.no_grad(), torch.cuda.amp.autocast():
                clip_features = clip_model.encode_image(clip_image)
                clip_features = torch.reshape(clip_features, (-1,))
                
                blip_tokenized_caption = clip_tokenizer(blip_caption)
                blip_text_features = clip_model.encode_text(blip_tokenized_caption)
                blip_text_features = torch.reshape(blip_text_features, (-1,))

            f_sim = torch.sigmoid(F.cosine_similarity(clip_features, blip_text_features, dim=0)).item()

            new_row = [r_sim, f_sim]
            tab_metrics.loc[len(tab_metrics)] = new_row

            print(tab_metrics)

            count += 1
        
    accuracy_path = os.path.join(outdir, "relations.csv")
    tab_metrics.to_csv(accuracy_path)

if __name__ == "__main__":
    encode_clip_blip()


