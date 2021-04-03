import sys
import os
from shutil import copyfile

def read_txt_list(set_path):
    with open(set_path) as f:
        set_list = f.readlines()
    return [i.strip() for i in set_list]

def main(basedir="/data/"):
    train_set_path = os.path.join(basedir, "sbd", "train.txt")
    val_set_path = os.path.join(basedir, "VOCdevkit", "VOC2012", "ImageSets", "Segmentation", "val.txt")
    assert os.path.exists(train_set_path), "SBD dataset is not downloaded!"
    assert os.path.exists(val_set_path), "VOC2012 is not downloaded!"
    voc_val_list = read_txt_list(val_set_path)
    sbd_train_list = read_txt_list(train_set_path)
    print("VOC Validation contains {} data points".format(len(voc_val_list)))
    print("SBD Train contains {} data points".format(len(sbd_train_list)))
    for train_img in sbd_train_list:
        try:
            idx = voc_val_list.index(train_img)
        except ValueError:
            continue
        # Intersect!
        voc_val_list.pop(idx)
    print("VOC2012 seg val (no intersect with SBD train) contains {} points".format(len(voc_val_list)))

    # There are two ways to do this
    #   1. remove intersection from VOC2012 seg val (done in FCN/OSLSM)
    #   2. remove intersection from SBD train
    # We follow previous works and use 1.
    with open("val.txt", 'w') as f:
        for val_img in voc_val_list:
            f.write(val_img)
            f.write('\n')
    key = input("Enter y to replace {}\n".format(val_set_path))
    if key.strip() == "y":
        # Back up original
        backup_path = val_set_path + ".bak"
        print("Backing up original Val label to {}".format(backup_path))
        copyfile(val_set_path, backup_path)
        # Copy
        copyfile("val.txt", val_set_path)

if __name__ == '__main__':
    main()