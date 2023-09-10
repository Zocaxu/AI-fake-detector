import torch
import torch.nn.functional as F
from PIL import Image
import open_clip
import os
import click
from tqdm import tqdm
import pandas as pd
import numpy as np
from models.FakeOrRealClassifier1 import FakeOrRealClassifier1
from models.FakeOrRealClassifier2 import FakeOrRealClassifier2
from models.FakeOrRealClassifier3 import FakeOrRealClassifier3

@click.command()
@click.argument("testdir", type=click.Path(exists=True))
@click.option("--outdir", default="./out_tests")
@click.option("--checkpoint", default=None, type=click.Path(exists=True))
@click.option("--feature_size", default=768)
def run_tests(testdir, outdir, checkpoint, feature_size):
    
    if checkpoint is None:
        print("Need a checkpoint to test.")
        exit(1)
    
    test_pts = os.listdir(testdir)

    print("===== Running tests on:")
    print(test_pts)

    print("===== Loading model...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = FakeOrRealClassifier1(feature_size)
    net.load_state_dict(torch.load(checkpoint))
    net.eval()
    net = net.to(device)
    sigmoid = torch.nn.Sigmoid()

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    thresholds = np.arange(start=0, stop=1.05, step=0.05)
    columns = ["{:.2f}".format(t) for t in thresholds]
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
            
            features = embed.to(device)
            pred = net(features)
            pred = sigmoid(pred)
            predictions[pt].append(pred)
    
    for tresh in tqdm(thresholds):
        total_fake = 0
        correct_fake = 0
        total_real = 0
        correct_real = 0
        for pt in test_pts:
            if pt in real_pts:
                filtered = list(filter(lambda score: score < tresh, predictions[pt]))
                total_real += len(predictions[pt])
                correct_real += len(filtered)
            else:
                filtered = list(filter(lambda score: score >= tresh, predictions[pt]))
                total_fake += len(predictions[pt])
                correct_fake += len(filtered)
            acc = len(filtered) / len(predictions[pt])
            tab_metrics.loc[pt, "{:.2f}".format(tresh)] = acc
        # Doing average like this is biaised, because test set sizes are not all balanced
        tab_metrics.loc["Real", "{:.2f}".format(tresh)] = (correct_real / total_real)
        tab_metrics.loc["Fake", "{:.2f}".format(tresh)] = (correct_fake / total_fake)
        tab_metrics.loc["Total","{:.2f}".format(tresh)] = ((correct_fake / total_fake) + (correct_real / total_real)) / 2
    
    accuracy_path = os.path.join(outdir, "accuracies.csv")
    tab_metrics.to_csv(accuracy_path)
    print(tab_metrics)




if __name__ == "__main__":
    run_tests()
