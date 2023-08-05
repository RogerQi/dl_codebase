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
        if os.path.exists('/root/autodl-tmp'):
            return "/root/autodl-tmp"
        elif os.path.exists('/data'):
            return "/data"
        elif os.path.exists('/scratch/bbqi/illinirm/data'):
            return "/scratch/bbqi/illinirm/data"
        else:
            raise Exception("Please specify dataset root by setting environment variable DATASET_ROOT")

def guess_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

