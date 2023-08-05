import __init_lib_path
from config_guard import cfg, update_config_from_yaml
import dataset
from tqdm import trange
import os
import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser(description = "Roger's Deep Learning Playground")
    parser.add_argument('--cfg', help = "specify particular yaml configuration to use", required = True,
        default = "configs/mnist_torch_official.yaml", type = str)
    parser.add_argument("--opts", help="Command line options to overwrite configs", default=[], nargs=argparse.REMAINDER)
    args = parser.parse_args()

    return args

def main():
    # --------------------------
    # | Initial set up
    # |  1. Parse arg and yaml
    # |  2. Set device
    # |  3. Set seed
    # --------------------------
    args = parse_args()
    update_config_from_yaml(cfg, args)

    # --------------------------
    # | Prepare datasets
    # --------------------------
    dataset_module = dataset.dataset_dispatcher(cfg)

    train_set = dataset_module.get_train_set(cfg)
    val_set = dataset_module.get_val_set(cfg)

    def create_mapping(my_set, my_name):
        # TODO: use fname to enforce class mapping consistency
        class_map = {}
        for i in trange(len(my_set)):
            img, label = my_set[(i, {'aug': False})]
            label = torch.unique(label)
            for l in label:
                l = int(l)
                if l in [0, -1, 255]: continue
                if l not in class_map:
                    class_map[l] = [str(i)]
                else:
                    class_map[l].append(str(i))
        # Write class_map
        class_map_dir = os.path.join('metadata', cfg.DATASET.dataset, my_name)
        assert not os.path.exists(class_map_dir)
        os.makedirs(class_map_dir)
        for c in class_map:
            with open(os.path.join(class_map_dir, str(c) + '.txt'), 'w') as f:
                f.write('\n'.join(class_map[c]))
    
    create_mapping(train_set, 'train')
    create_mapping(val_set, 'val')
    
if __name__ == '__main__':
    main()
