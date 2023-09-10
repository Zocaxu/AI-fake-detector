import torch
import torch.nn.functional as F
from PIL import Image
import open_clip
import os
import click
from tqdm import tqdm
from models.FakeOrRealClassifier import FakeOrRealClassifier

def load_model(model_name, pretrain):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrain)
    return model, preprocess

@click.command()
@click.argument("testdir", type=click.Path(exists=True))
@click.option("--bank_real", default="./feature_bank")
@click.option("--bank_fake", default="./feature_bank")
@click.option("--outdir", default="./out_tests")
@click.option("--knn", default=9)
@click.option("--model", default="ViT-L-14")
@click.option("--pretrain", default="datacomp_xl_s13b_b90k")
@click.option("--onedir", default=None)
@click.option("--method", default="knn")
@click.option("--checkpoint", default=None, type=click.Path(exists=True))
def run_tests(testdir, bank_real, bank_fake, outdir, knn, model, pretrain, onedir, method, checkpoint):
    
    methods = ["knn", "net"]
    if method not in methods:
        print(f"Method \"{method}\" doesn't exist, please select method in {methods}")
        exit(1)
    
    if onedir is not None:
        test_dirs = [onedir]
    else:
        test_dirs = os.listdir(testdir)

    if not os.path.exists(outdir):
        os.mkdir(outdir)
        os.mkdir(os.path.join(outdir, "predictions"))
        os.mkdir(os.path.join(outdir, "accuracies"))

    print("===== Running tests on:")
    print(test_dirs)

    print("===== Loading model...")
    model, preprocess = load_model(model, pretrain)

    if method == "knn":
        fake_features = torch.load(os.path.join(bank_fake, "fake_features"))
        real_features = torch.load(os.path.join(bank_real, "real_features"))
        size = min(fake_features.shape[0], real_features.shape[0])
        fake_features = fake_features[:size]
        real_features = real_features[:size]
        print("Done. Shape of bank: ", size)
    
    if method == "net":
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        net = FakeOrRealClassifier(768)
        net.load_state_dict(torch.load(checkpoint))
        net.eval()
        net = net.to(device)
        sigmoid = torch.nn.Sigmoid()

    for dir in test_dirs:
        predfile = open(os.path.join(outdir, "predictions", dir), "w")
        try:
            print("Test directory:", dir)
            n_reals = 0
            n_fakes = 0
            for file in tqdm(os.listdir(os.path.join(testdir, dir))):
                file_path = os.path.join(testdir, dir, file)
                image = preprocess(Image.open(file_path)).unsqueeze(0)
                with torch.no_grad(), torch.cuda.amp.autocast():
                    image_features = model.encode_image(image)

                if method == "knn":
                    fake_cosine_sim = torch.topk(F.cosine_similarity(fake_features, image_features),knn//2 + 1,sorted=True)
                    real_cosine_sim = torch.topk(F.cosine_similarity(real_features, image_features),knn//2 + 1,sorted=True)
                    if fake_cosine_sim.values[-1] >= real_cosine_sim.values[-1]:
                        n_fakes += 1
                    else:
                        n_reals += 1
                
                if method == "net":
                    features = image_features.to(device)
                    pred = net(features)
                    predfile.write(f"{file}, {pred.item()}, {True if pred >= 0 else False}\n")
                    pred_digit = sigmoid(pred).round()
                    if pred_digit == 1:
                        n_fakes += 1
                    else:
                        n_reals += 1

            print(f"Detected {n_fakes} fakes and {n_reals} reals ({n_fakes/(n_reals+n_fakes)})")
            retfile = open(os.path.join(outdir, "accuracies", dir), "w")
            retfile.write(f"{dir} : {n_fakes} / {n_reals} ({n_fakes/(n_reals+n_fakes)})")
            retfile.close()
            predfile.close()
        except :
            predfile.close()
            continue



if __name__ == "__main__":
    run_tests()
