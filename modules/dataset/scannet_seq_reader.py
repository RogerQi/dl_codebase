from PIL import Image

import os, struct
import numpy as np
import zlib
import imageio
import cv2
import csv
import shutil

from tqdm import tqdm

import zipfile

root_dir = "/media/eason/My Passport/data/scannet_v2"

label_file = os.path.join(root_dir, 'scannetv2-labels.combined.tsv')

def unzip(zip_path, zip_type):
    assert zip_type in ["instance-filt", "label-filt"]
    target_dir = f'/tmp/{zip_type}'
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    return os.path.join(target_dir, zip_type)

scannet_id_nyu_dict = {}

with open(label_file, newline='') as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t')
    for row_dict in reader:
        scannet_id = row_dict['id']
        nyu40_id = row_dict['nyu40id']
        scannet_id_nyu_dict[int(scannet_id)] = int(nyu40_id)

# map label as instructed in http://kaldir.vc.in.tum.de/scannet_benchmark/labelids.txt
VALID_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])

scannet_subset_map = np.zeros(41) # NYU40 has 40 labels
for i in range(len(VALID_CLASS_IDS)):
    scannet_subset_map[VALID_CLASS_IDS[i]] = i + 1

# This dict maps from fine-grained ScanNet ids (579 categories)
# to the 20 class subset as in the benchmark
scannet_mapping = np.zeros(max(scannet_id_nyu_dict) + 1)

for k in scannet_id_nyu_dict:
    scannet_mapping[k] = scannet_subset_map[scannet_id_nyu_dict[k]]

# HARDCODE FOR NOW
printer_scannet_id = 50

scannet_mapping[printer_scannet_id] = len(VALID_CLASS_IDS) + 1

COMPRESSION_TYPE_COLOR = {-1:'unknown', 0:'raw', 1:'png', 2:'jpeg'}
COMPRESSION_TYPE_DEPTH = {-1:'unknown', 0:'raw_ushort', 1:'zlib_ushort', 2:'occi_ushort'}

class RGBDFrame():
    def load(self, file_handle):
        self.camera_to_world = np.asarray(struct.unpack('f'*16, file_handle.read(16*4)), dtype=np.float32).reshape(4, 4)
        self.timestamp_color = struct.unpack('Q', file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack('Q', file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.color_data = b''.join(struct.unpack('c'*self.color_size_bytes, file_handle.read(self.color_size_bytes)))
        self.depth_data = b''.join(struct.unpack('c'*self.depth_size_bytes, file_handle.read(self.depth_size_bytes)))

    def decompress_depth(self, compression_type):
        if compression_type == 'zlib_ushort':
             return self.decompress_depth_zlib()
        else:
             raise

    def decompress_depth_zlib(self):
        return zlib.decompress(self.depth_data)

    def decompress_color(self, compression_type):
        if compression_type == 'jpeg':
             return self.decompress_color_jpeg()
        else:
             raise

    def decompress_color_jpeg(self):
        return imageio.imread(self.color_data)

class scannet_scene_reader:
    def __init__(self, root_dir, scene_name):
        self.version = 4
        
        # Get file paths
        sens_path = os.path.join(root_dir, 'scans', scene_name, f'{scene_name}.sens')
        semantic_zip_path = os.path.join(root_dir, 'scans', scene_name, f'{scene_name}_2d-label-filt.zip')
        instance_zip_path = os.path.join(root_dir, 'scans', scene_name, f'{scene_name}_2d-instance-filt.zip')
        
        # Load
        self.load(sens_path)
        self.label_dir = unzip(semantic_zip_path, 'label-filt')
        self.inst_dir = unzip(instance_zip_path, 'instance-filt')

    def load(self, filename):
        with open(filename, 'rb') as f:
            # Read meta data
            version = struct.unpack('I', f.read(4))[0]
            assert self.version == version
            strlen = struct.unpack('Q', f.read(8))[0]
            self.sensor_name = b''.join(struct.unpack('c'*strlen, f.read(strlen)))
            self.intrinsic_color = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
            self.extrinsic_color = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
            self.intrinsic_depth = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
            self.extrinsic_depth = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack('i', f.read(4))[0]]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack('i', f.read(4))[0]]
            self.color_width = struct.unpack('I', f.read(4))[0]
            self.color_height =    struct.unpack('I', f.read(4))[0]
            self.depth_width = struct.unpack('I', f.read(4))[0]
            self.depth_height =    struct.unpack('I', f.read(4))[0]
            self.depth_shift =    struct.unpack('f', f.read(4))[0]
            num_frames =    struct.unpack('Q', f.read(8))[0]
            
            # Read frames
            self.frames = []
            for i in tqdm(range(num_frames)):
                frame = RGBDFrame()
                frame.load(f)
                self.frames.append(frame)
    
    def __getitem__(self, idx):
        # TODO: use dynamic image size
        image_size = (480, 640)
        assert idx >= 0
        assert idx < len(self.frames)
        depth_data = self.frames[idx].decompress_depth(self.depth_compression_type)
        depth = np.fromstring(depth_data, dtype=np.uint16).reshape(self.depth_height, self.depth_width)
        if image_size is not None:
            depth = cv2.resize(depth, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
        color = self.frames[idx].decompress_color(self.color_compression_type)
        if image_size is not None:
            color = cv2.resize(color, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
        pose = self.frames[idx].camera_to_world
        
        # Read label
        label_path = os.path.join(self.label_dir, f"{idx}.png")
        label_map = np.array(Image.open(label_path))
        if image_size is not None:
            label_map = cv2.resize(label_map, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
        label_map = scannet_mapping[label_map]
        
        # Read instance map
        inst_path = os.path.join(self.inst_dir, f"{idx}.png")
        inst_map = np.array(Image.open(inst_path))
        if image_size is not None:
            inst_map = cv2.resize(inst_map, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
        
        return {
            'color': color,
            'depth': depth,
            'pose': pose,
            'intrinsics_color': self.intrinsic_color,
            'semantic_label': label_map,
            'inst_label': inst_map
        }
    
    def __len__(self):
        return len(self.frames)

