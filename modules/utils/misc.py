import os
import torch
import urllib

def download_file(url, local_path):
    g = urllib.request.urlopen(url)
    with open(local_path, 'b+w') as f:
        f.write(g.read())

def get_dataset_root():
    try:
        return os.environ['DATASET_ROOT']
    except KeyError:
        return "/data"

def guess_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
