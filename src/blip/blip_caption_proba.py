from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip_itm import blip_itm

import os
from tqdm import tqdm
import click
import open_clip
import numpy as np
import csv
import pandas as pd

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
@click.argument("blip_dir", type=click.Path(exists=True))
@click.argument("coco_dir", type=click.Path(exists=True))
@click.argument("coco_csv", type=click.Path(exists=True))
@click.option("--outdir", default="./out")
@click.option("--blip_checkpoint", default=None)
@click.option("--blip_img_size", default=512)
@click.option("--blip_mode", default="large")
def encode_clip_blip(diffusion_dir,blip_dir, coco_dir, coco_csv, outdir, blip_checkpoint, blip_img_size, blip_mode):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    blip_model = blip_itm(
        pretrained=blip_checkpoint,
        image_size=blip_img_size,
        vit=blip_mode,
        med_config="/vol/bitbucket/mb322/BLIP/configs/med_config.json")
    blip_model.eval()
    blip_model = blip_model.to(device)

    if(not os.path.exists(outdir)):
        os.makedirs(outdir)

    tab = ({
        'File': ["Test"],
        'Real-Natural':[0],
        'Real-BLIP' :[0],
        'Fake-Natural': [0],
        'Fake-BLIP': [0]
        })
    tab_metrics = pd.DataFrame(tab)
    print(tab_metrics)

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
            blip_image = load_image_for_blip(img_path,device,512)
            with torch.no_grad(), torch.cuda.amp.autocast():
                itm_output = blip_model(blip_image, coco_caption, match_head='itm')
                real_natural = torch.nn.functional.softmax(itm_output,dim=1)[:,1]

                real_blip_path = os.path.join(blip_dir, "real", "blip", "captions", coco_file_name + '.txt')
                with open(real_blip_path, 'r') as cap_file:
                    blip_caption = cap_file.readline()
                itm_output = blip_model(blip_image, blip_caption, match_head='itm')
                real_blip = torch.nn.functional.softmax(itm_output,dim=1)[:,1]
            
            # Stable diffusion image
            stable_file_name = "{:05d}".format(count) + ".png"
            img_path = os.path.join(diffusion_dir, stable_file_name)
            blip_image = load_image_for_blip(img_path,device,512)

            with torch.no_grad(), torch.cuda.amp.autocast():
                itm_output = blip_model(blip_image, coco_caption, match_head='itm')
                fake_natural = torch.nn.functional.softmax(itm_output,dim=1)[:,1]

                real_blip_path = os.path.join(blip_dir, "fake", "blip", "captions", stable_file_name + '.txt')
                with open(real_blip_path, 'r') as cap_file:
                    blip_caption = cap_file.readline()
                itm_output = blip_model(blip_image, blip_caption, match_head='itm')
                fake_blip = torch.nn.functional.softmax(itm_output,dim=1)[:,1]


            new_row = [coco_file_name, real_natural.item(), real_blip.item(), fake_natural.item(), fake_blip.item()]
            tab_metrics.loc[len(tab_metrics)] = new_row
            
            count += 1
        
    accuracy_path = os.path.join(outdir, "accuracies.csv")
    tab_metrics.to_csv(accuracy_path)

if __name__ == "__main__":
    encode_clip_blip()


