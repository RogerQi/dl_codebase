import os

def get_dataset_root():
    try:
        return os.environ['DATASET_ROOT']
    except KeyError:
        return "/data"
