import os
from PIL import Image
import tqdm
import random
import csv
import click
from tqdm import tqdm

resize_range = (50, 200)
qf_range = (65, 100)

def check_img(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif'))

@click.command()
@click.argument("testdir")
@click.argument("outdir")
def create_csv_operation(testdir, outdir):
    test_sets = os.listdir(testdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    with open(os.path.join(outdir, "operations.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["file","qf","resize"])
        for test_set in test_sets:
            if not os.path.isdir(os.path.join(testdir, test_set)):
                continue
            files = os.listdir(os.path.join(testdir, test_set))
            for file in tqdm(files):
                if not check_img(file):
                    continue
                qf = random.randint(qf_range[0], qf_range[1])
                resize = random.randint(resize_range[0], resize_range[1])
                writer.writerow([
                    test_set + '/' + file,
                    qf,
                    resize
                ])



if __name__ == "__main__":
    create_csv_operation()