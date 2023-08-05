import os
import numpy as np

# requires LVIS api from https://github.com/lvis-dataset/lvis-api

import lvis
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

def single_pass_one_split(base_dir, train):
    if train:
        ann_path = os.path.join(base_dir, 'lvis_v1_train.json')
        mode_str = 'train'
    else:
        ann_path = os.path.join(base_dir, 'lvis_v1_val.json')
        mode_str = 'val'

    lvis_api = lvis.LVIS(ann_path)

    img_ids = sorted(lvis_api.imgs.keys())

    imgs_metainfo_list = lvis_api.load_imgs(img_ids)

    # Target DIR
    metadata_dir = 'metadata/lvis_150'
    seg_mask_dir = os.path.join(base_dir, 'lvis_150_masks')

    train_seg_mask_dir = os.path.join(seg_mask_dir, 'train2017')
    val_seg_mask_dir = os.path.join(seg_mask_dir, 'val2017')

    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(seg_mask_dir, exist_ok=True)
    os.makedirs(train_seg_mask_dir, exist_ok=True)
    os.makedirs(val_seg_mask_dir, exist_ok=True)

    freq_cnt_dict = {'f': [], 'c': [], 'r': []}

    for i in range(1, 1204):
        cat_info = lvis_api.load_cats(ids=[i])[0]
        cat_name = cat_info['name']
        cat_img_cnt = cat_info['image_count']
        cat_id = cat_info['id']
        freq = cat_info['frequency']
        freq_cnt_dict[freq].append((cat_name, cat_img_cnt, cat_id))

    for k in freq_cnt_dict:
        freq_cnt_dict[k] = sorted(freq_cnt_dict[k], key=lambda x: x[0].lower())

    final_classes_dict = {}
    class_per_freq = 50

    for k in freq_cnt_dict:
        assert class_per_freq < len(freq_cnt_dict[k])
        idx_list = np.linspace(0, len(freq_cnt_dict[k]), num=class_per_freq, endpoint=False).astype(int)
        final_classes_dict[k] = []
        for idx in idx_list:
            final_classes_dict[k].append(freq_cnt_dict[k][idx])
    
    # Create base_novel map for convenience
    novel_class_idx = np.arange(1, 120 + 30 + 1, 5)
    novel_class_idx = list(novel_class_idx)
    base_class_idx = [i for i in range(1, 151) if i not in novel_class_idx]

    # Create class map
    visible_class_list = []

    for k in ['f', 'c', 'r']:
        for obj_name, img_cnt, cat_id in final_classes_dict[k]:
            visible_class_list.append((obj_name, cat_id))

    class_map = np.zeros((1204))

    for i in range(len(visible_class_list)):
        vanilla_id = visible_class_list[i][1]
        mapped_idx = i + 1
        if mapped_idx in base_class_idx:
            final_mapped_idx = base_class_idx.index(mapped_idx) + 1
        else:
            final_mapped_idx = novel_class_idx.index(mapped_idx) + 1 + len(base_class_idx)
        class_map[vanilla_id] = final_mapped_idx
    
    # print sorted class names
    class_map_list = list(class_map)

    names_list = []

    for i in range(1, 151):
        vanilla_idx = class_map_list.index(i)
        obj_name = lvis_api.load_cats([vanilla_idx])[0]['name']
        names_list.append(obj_name)

    print(names_list)

    valid_img_ids = []

    for img_id in tqdm(img_ids):
        # Build mask
        ann_list = lvis_api.img_ann_map[img_id]
        split_folder, file_name = lvis_api.load_imgs([img_id])[0]['coco_url'].split("/")[-2:]
        seg_mask = None
        for ann in ann_list:
            class_id = ann['category_id']
            if class_map[class_id] == 0:
                continue
            else:
                real_class_id = class_map[class_id]
            ann_mask = torch.from_numpy(lvis_api.ann_to_mask(ann))
            if seg_mask is None:
                seg_mask = torch.zeros_like(ann_mask, dtype=torch.int)
            seg_mask = torch.max(seg_mask, ann_mask * real_class_id)
        if seg_mask is None:
            continue
        else:
            assert seg_mask.max() < 256 and seg_mask.min() >= 0
            valid_img_ids.append(str(img_id) + ' ' + split_folder + ' ' + file_name)
            save_path = os.path.join(seg_mask_dir, split_folder, file_name.replace('.jpg', '.png'))
            seg_mask = np.array(seg_mask).astype(np.uint8)
            Image.fromarray(seg_mask).save(save_path)
        
    # Write img_ids to metadata
    with open(os.path.join(metadata_dir, f'{mode_str}.txt'), 'w') as f:
        f.write('\n'.join(valid_img_ids))

if __name__ == '__main__':
    base_dir = '/data/COCO2017'
    for train in [True, False]:
        single_pass_one_split(base_dir, train)
