import os
import click
from tqdm import tqdm
import numpy as np

DATADIR = "/vol/bitbucket/mb322/Downloads/mscoco512"
OUTDIR = "/vol/bitbucket/mb322/AI-fake-detection/mscoco_prompts/"
N = 5000

if(not os.path.exists(OUTDIR)):
    os.mkdir(OUTDIR)
gather_file_txt = open(os.path.join(OUTDIR, f"gather_mscoco_{N}_10.txt"), "w")
gather_file_csv = open(os.path.join(OUTDIR, f"gather_mscoco_{N}_10.csv"), "w")

indexes = np.random.choice(500000,N)
for n in tqdm(indexes):
    q = n // 10000
    folder = "{:05d}".format(q)
    text_file = "{:09d}".format(n) + ".txt"
    img_file = "{:09d}".format(n) + ".jpg"
    try:
        f = open(os.path.join(DATADIR, folder, text_file), "r")
    except:
        print(f"Error: cannot open file at location {os.path.join(DATADIR, folder, text_file)}")
        continue
    prompt = f.readlines()[0]
    if prompt[-1] == '\n':
        prompt = prompt[0:-1]
    gather_file_csv.write(img_file + "," + prompt + '\n')
    gather_file_txt.write(prompt + '\n')
    f.close()
gather_file_txt.close()
gather_file_csv.close()