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
@click.argument("progan_dir", type=click.Path(exists=True))
@click.option("--n_total", default=200000)
@click.option("--outdir", default="./out")
@click.option("--clip_model", default="ViT-L-14")
@click.option("--clip_pretrain", default="datacomp_xl_s13b_b90k")
@click.option("--blip_checkpoint", default=None)
@click.option("--blip_img_size", default=512)
@click.option("--blip_mode", default="large")
def encode_clip_blip(progan_dir, n_total, outdir, clip_model, clip_pretrain, blip_checkpoint, blip_img_size, blip_mode):
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
        os.makedirs(os.path.join(outdir, "real", "blip", "captions"))
        os.makedirs(os.path.join(outdir, "real", "blip", "features"))

    N = n_total // 40
    count = 0

    for subdir, dirs, files in os.walk(progan_dir):
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
            img_path = os.path.join(subdir,file)
            image = Image.open(img_path)

            clip_image = clip_preprocess(image).unsqueeze(0)
            blip_image = load_image_for_blip(img_path,device,512)

            with torch.no_grad(), torch.cuda.amp.autocast():
                clip_features = clip_model.encode_image(clip_image)
                clip_features = torch.reshape(clip_features, (-1,))
                
                blip_caption = blip_model.generate(blip_image, sample=False, num_beams=3, max_length=20, min_length=5)[0] 
                blip_tokenized_caption = clip_tokenizer(blip_caption)
                blip_text_features = clip_model.encode_text(blip_tokenized_caption)
                blip_text_features = torch.reshape(blip_text_features, (-1,))
            
            blip_cat_features = torch.cat((clip_features,blip_text_features))
            
            torch.save(blip_cat_features, os.path.join(outdir, label, "blip", "features", str(file) + ".pt"))

            with open(os.path.join(outdir, label, "blip", "captions", str(file) + ".txt"), 'w') as cap_file:
                cap_file.write(blip_caption)
            
            count += 1

if __name__ == "__main__":
    encode_clip_blip()


