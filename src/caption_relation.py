import torch
import torch.nn.functional as F
import numpy as np
import os
import csv
import pandas as pd
import click
from tqdm import tqdm

@click.command()
@click.argument("dir")
@click.argument("outdir")
def caption_relation(dir, outdir):
    fake_features = torch.load(os.path.join(dir, "fake_features"))
    real_features = torch.load(os.path.join(dir, "real_features"))

    if(not os.path.exists(outdir)):
        os.makedirs(outdir)

    tab = ({
        'Real-BLIP' :[0],
        'Fake-BLIP': [0]
        })
    tab_metrics = pd.DataFrame(tab)

    N = min(fake_features.shape[0], real_features.shape[0])

    for i in tqdm(range(N)):
        f_f = fake_features[i]
        split = int(f_f.shape[0]/2)
        f_clip_embed = f_f[:split]
        f_blip_embed = f_f[split:]
        f_sim = torch.sigmoid(F.cosine_similarity(f_clip_embed, f_blip_embed, dim=0)).item()

        r_f = real_features[i]
        split = int(r_f.shape[0]/2)
        r_clip_embed = r_f[:split]
        r_blip_embed = r_f[split:]
        r_sim = torch.sigmoid(F.cosine_similarity(r_clip_embed, r_blip_embed, dim=0)).item()

        new_row = [r_sim, f_sim]
        tab_metrics.loc[len(tab_metrics)] = new_row
    
    accuracy_path = os.path.join(outdir, "relations.csv")
    tab_metrics.to_csv(accuracy_path)



if __name__ == "__main__":
    caption_relation()