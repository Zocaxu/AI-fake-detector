from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip import blip_decoder

import os
from tqdm import tqdm
import click
import open_clip
import numpy as np
import csv

def load_model(model_name, pretrain):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrain)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer

def load_image_for_blip(img_path, device, image_size=512):
    raw_image = Image.open(img_path).convert("RGB")

    w,h = raw_image.size
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        ]) 
    image = transform(raw_image).unsqueeze(0).to(device)   
    return image

@click.command()
@click.argument("testdir", type=click.Path(exists=True))
@click.option("--outdir", default="./out")
@click.option("--clip_model", default="ViT-L-14")
@click.option("--clip_pretrain", default="datacomp_xl_s13b_b90k")
@click.option("--blip_checkpoint", default=None)
@click.option("--blip_img_size", default=512)
@click.option("--blip_mode", default="large")
@click.option("--feature_size", default=1536)
@click.option("--one_dir", default=None)
def encode_clip_blip(testdir, outdir,clip_model,clip_pretrain, blip_checkpoint, blip_img_size, blip_mode, feature_size, one_dir):
    print("===== Loading models...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_model, clip_preprocess, clip_tokenizer = load_model(clip_model, clip_pretrain)
    blip_model = blip_decoder(
        pretrained=blip_checkpoint,
        image_size=blip_img_size, 
        vit=blip_mode, 
        med_config="/vol/bitbucket/mb322/BLIP/configs/med_config.json")
    blip_model.eval()
    blip_model = blip_model.to(device)

    if(not os.path.exists(outdir)):
        os.mkdir(outdir)
    
    if one_dir is not None:
        test_dirs = [one_dir]
    else:
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
                img_path = os.path.join(testdir, dir, file)
                image = Image.open(img_path)
                clip_image = clip_preprocess(image).unsqueeze(0)
                blip_image = load_image_for_blip(img_path, device, 512)

                with torch.no_grad(), torch.cuda.amp.autocast():
                    clip_features = clip_model.encode_image(clip_image)
                    clip_features = torch.reshape(clip_features, (-1,))
                    
                    blip_caption = blip_model.generate(blip_image, sample=False, num_beams=3, max_length=20, min_length=5)[0] 
                    blip_tokenized_caption = clip_tokenizer(blip_caption)
                    blip_text_features = clip_model.encode_text(blip_tokenized_caption)
                    blip_text_features = torch.reshape(blip_text_features, (-1,))

                    blip_cat_features = torch.cat((clip_features,blip_text_features))
                
                embeddings[count] = blip_cat_features
                count += 1
            
            torch.save(embeddings, os.path.join(outdir, dir + '.pt'))
        except:
            print(f"error reading {dir}")
            continue

if __name__ == "__main__":
    encode_clip_blip()


