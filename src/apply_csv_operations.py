import os
from PIL import Image
import tqdm
import random
import csv
import click
from tqdm import tqdm

def check_img(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif'))

@click.command()
@click.argument("input_dir")
@click.argument("csv_file")
@click.argument("output_dir")
def create_csv_operation(input_dir, csv_file, output_dir):
    print('CSV Operations from ', input_dir, 'to', output_dir, flush=True)

    with open(csv_file, 'r') as csvfile:
        data = list(csv.reader(csvfile, delimiter=',', quotechar='|'))
        for i in tqdm(range(1, len(data))):
            row = data[i]
            # open image
            img = Image.open(os.path.join(input_dir, row[0])).convert('RGB')

            # read the jpeg quality factor
            qf = int(row[1])

            # read resizing factor
            rf = int(row[2])

            # select interpolation
            interp = Image.BICUBIC

            x, y = img.size
            img = img.resize((int(x * rf / 100), int(y * rf / 100)), interp)
            
            path_split = row[0].split('/')
            folder = path_split[0]
            if not os.path.exists(os.path.join(output_dir, folder)):
                os.mkdir(os.path.join(output_dir, folder))
            dst = os.path.join(os.path.join(output_dir, folder), path_split[1])

            img.save(dst, "JPEG", quality=qf)


if __name__ == "__main__":
    create_csv_operation()