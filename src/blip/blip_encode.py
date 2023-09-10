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
@click.argument("diffusion_dir", type=click.Path(exists=True))
@click.argument("coco_dir", type=click.Path(exists=True))
@click.argument("coco_csv", type=click.Path(exists=True))
@click.option("--outdir", default="./out")
@click.option("--clip_model", default="ViT-L-14")
@click.option("--clip_pretrain", default="datacomp_xl_s13b_b90k")
@click.option("--blip_checkpoint", default=None)
@click.option("--blip_img_size", default=512)
@click.option("--blip_mode", default="large")
def encode_clip_blip(diffusion_dir,coco_dir,coco_csv,outdir,clip_model,clip_pretrain, blip_checkpoint, blip_img_size, blip_mode):
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
        os.makedirs(os.path.join(outdir, "fake", "blip", "captions"))
        os.makedirs(os.path.join(outdir, "fake", "blip", "features"))
        os.makedirs(os.path.join(outdir, "fake", "natural", "captions"))
        os.makedirs(os.path.join(outdir, "fake", "natural", "features"))
        os.makedirs(os.path.join(outdir, "real", "blip", "captions"))
        os.makedirs(os.path.join(outdir, "real", "blip", "features"))
        os.makedirs(os.path.join(outdir, "real", "natural", "captions"))
        os.makedirs(os.path.join(outdir, "real", "natural", "features"))

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
            blip_image = load_image_for_blip(img_path,device,512)
            with torch.no_grad(), torch.cuda.amp.autocast():
                clip_features = clip_model.encode_image(clip_image)
                clip_features = torch.reshape(clip_features, (-1,))
                
                blip_caption = blip_model.generate(blip_image, sample=False, num_beams=3, max_length=20, min_length=5)[0] 
                blip_tokenized_caption = clip_tokenizer(blip_caption)
                blip_text_features = clip_model.encode_text(blip_tokenized_caption)
                blip_text_features = torch.reshape(blip_text_features, (-1,))

                natural_caption = coco_caption
                natural_tokenized_caption = clip_tokenizer(natural_caption)
                natural_text_features = clip_model.encode_text(natural_tokenized_caption)
                natural_text_features = torch.reshape(natural_text_features, (-1,))
            
            blip_cat_features = torch.cat((clip_features,blip_text_features))
            natural_cat_features = torch.cat((clip_features,natural_text_features))
            
            torch.save(blip_cat_features, os.path.join(outdir, "real", "blip", "features", coco_file_name + ".pt"))
            torch.save(natural_cat_features, os.path.join(outdir, "real", "natural", "features", coco_file_name + ".pt"))

            with open(os.path.join(outdir,"real", "blip", "captions", coco_file_name + ".txt"), 'w') as cap_file:
                cap_file.write(blip_caption)
            with open(os.path.join(outdir,"real", "natural", "captions", coco_file_name + ".txt"), 'w') as cap_file:
                cap_file.write(coco_caption)
            
            # Stable diffusion image
            stable_file_name = "{:05d}".format(count) + ".png"
            img_path = os.path.join(diffusion_dir, stable_file_name)
            stable_image = Image.open(os.path.join(diffusion_dir, stable_file_name))
            clip_image = clip_preprocess(stable_image).unsqueeze(0)
            blip_image = load_image_for_blip(img_path,device,512)

            with torch.no_grad(), torch.cuda.amp.autocast():
                clip_features = clip_model.encode_image(clip_image)
                clip_features = torch.reshape(clip_features, (-1,))
                
                blip_caption = blip_model.generate(blip_image, sample=False, num_beams=3, max_length=20, min_length=5)[0] 
                blip_tokenized_caption = clip_tokenizer(blip_caption)
                blip_text_features = clip_model.encode_text(blip_tokenized_caption)
                blip_text_features = torch.reshape(blip_text_features, (-1,))

                natural_caption = coco_caption
                natural_tokenized_caption = clip_tokenizer(natural_caption)
                natural_text_features = clip_model.encode_text(natural_tokenized_caption)
                natural_text_features = torch.reshape(natural_text_features, (-1,))
            
            blip_cat_features = torch.cat((clip_features,blip_text_features))
            natural_cat_features = torch.cat((clip_features,natural_text_features))
            print(blip_cat_features.shape)
            
            torch.save(blip_cat_features, os.path.join(outdir, "fake", "blip", "features", stable_file_name + ".pt"))
            torch.save(natural_cat_features, os.path.join(outdir, "fake", "natural", "features", stable_file_name + ".pt"))

            with open(os.path.join(outdir,"fake", "blip", "captions", stable_file_name + ".txt"), 'w') as cap_file:
                cap_file.write(blip_caption)
            with open(os.path.join(outdir,"fake", "natural", "captions", stable_file_name + ".txt"), 'w') as cap_file:
                cap_file.write(coco_caption)

            count += 1

if __name__ == "__main__":
    encode_clip_blip()


