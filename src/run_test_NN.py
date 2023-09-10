import torch
import torch.nn.functional as F
from PIL import Image
import open_clip
import os
import click
from tqdm import tqdm
import pandas as pd
import numpy as np
from models.FakeOrRealClassifier import FakeOrRealClassifier
from models.FakeOrRealClassifier2 import FakeOrRealClassifier2

@click.command()
@click.argument("testdir", type=click.Path(exists=True))
@click.option("--outdir", default="./out_tests")
@click.option("--bank_fake", default=None, type=click.Path(exists=True))
@click.option("--bank_real", default=None, type=click.Path(exists=True))
@click.option("--knn", default=9)
def run_tests(testdir, outdir, bank_fake, bank_real, knn):
    
    if bank_fake is None or bank_real is None:
        print("Need a bank to test.")
        exit(1)
    
    test_pts = os.listdir(testdir)

    print("===== Running tests on:")
    print(test_pts)

    print("===== Loading model...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    fake_features = torch.load(bank_fake)
    real_features = torch.load(bank_real)
    size = min(fake_features.shape[0], real_features.shape[0])
    fake_features = fake_features[:size]
    real_features = real_features[:size]
    print("Done. Shape of bank: ", size*2)

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    columns = ["Acc."]
    indexes = os.listdir(testdir)
    indexes.append("Real")
    indexes.append("Fake")
    indexes.append("Total")
    tab_metrics = pd.DataFrame(index=indexes, columns=columns)

    real_pts = ['real_coco_valid.pt', 'real_imagenet_val.pt', 'real_ucid.pt', 'celebA.pt', 'ffhq.pt']

    predictions = {}

    total_accuracy = []

    for pt in test_pts:
        print("Test directory:", pt)
        n_reals = 0
        n_fakes = 0

        embeddings = torch.load(os.path.join(testdir, pt))
        N = len(embeddings)
        
        predictions[pt] = []

        for i in tqdm(range(N)):
            embed = embeddings[i]

            fake_cosine_sim = torch.topk(F.cosine_similarity(fake_features, embed),knn//2 + 1,sorted=True)
            real_cosine_sim = torch.topk(F.cosine_similarity(real_features, embed),knn//2 + 1,sorted=True)
        
            if fake_cosine_sim.values[-1] >= real_cosine_sim.values[-1]:
                n_fakes += 1 
            else:
                n_reals += 1
        
        if pt in real_pts:
            acc = n_reals / (n_fakes + n_reals)
        else:
            acc = n_fakes / (n_fakes + n_reals)
            
        tab_metrics.loc[pt, "Acc."] = "{:.2f}".format(acc)
    
    accuracy_path = os.path.join(outdir, "accuracies.csv")
    tab_metrics.to_csv(accuracy_path)
    print(tab_metrics)


if __name__ == "__main__":
    run_tests()
