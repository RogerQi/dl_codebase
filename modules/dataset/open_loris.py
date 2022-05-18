import os
import numpy as np
from PIL import Image
import re
from functools import partial

import os
import numpy as np
from PIL import Image
from torchvision import datasets, transforms

class OpenLORIS_single_sequence_reader(datasets.vision.VisionDataset):
    def __init__(self, root, object_name, factor, difficulty):
        super(OpenLORIS_single_sequence_reader, self).__init__(root, None, None, None)
        assert difficulty in [0, 1, 2]
        self.object_name = object_name
        self.factor_name = factor
        # Get name of corresponding difficulty level
        self.diff_level_num = difficulty
        self.diff_level_name = self.factor_difficulty_check(factor, difficulty)
        
        # Get mask/bbox annotation dir of this video stream
        self.annotation_dir, self.scene_num = self.search_mask_folder(self.root, self.factor_name, self.diff_level_name, self.object_name)
        assert len(os.listdir(self.annotation_dir)) % 3 == 0 # visualized, mask, bbox coordinate txt
        self.sequence_len = len(os.listdir(self.annotation_dir)) // 3
        
        # Make a sorted list of mask PNGs
        self.seg_mask_path_list = []
        for fn in os.listdir(self.annotation_dir):
            if 'show' in fn: continue # skip visualization
            if fn.endswith('.txt'): continue # skip bbox txt
            assert self.get_image_number(fn, fn_suffix='.png') >= 0 # check fn format
            self.seg_mask_path_list.append(os.path.join(self.annotation_dir, fn))
        self.seg_mask_path_list = sorted(self.seg_mask_path_list,
                                         key=partial(self.get_image_number, fn_suffix='.png'))
        assert len(self.seg_mask_path_list) == self.sequence_len
        
        # indices of video frames with mask annotations available
        # some video sequence is malformed and not all frames are given segmentation mask...
        seg_mask_indices = [self.get_image_number(i, fn_suffix='.png') for i in self.seg_mask_path_list]
        
        # Aggregate raw frames from original train/val/test sets of OpenLORIS
        potential_segments = [f'segment{i + 3 * difficulty + 1}' for i in range(3)]
        sequence_frame_list = []
        for split in ['train', 'validation', 'test']:
            factor_base_path = os.path.join(self.root, split, self.factor_name)
            for segment_name in potential_segments:
                search_dir = os.path.join(factor_base_path, segment_name, self.object_name)
                assert os.path.exists(search_dir)
                potential_frames_list = os.listdir(search_dir)
                # simple sanity check
                for fn in potential_frames_list:
                    assert self.get_image_number(fn) >= 0
                sequence_frame_list += [os.path.join(search_dir, i) for i in potential_frames_list]
        
        # Make a sorted list of raw RGB images
        # in some sequence, there are more raw RGB frames than annotated frames.
        assert len(sequence_frame_list) >= self.sequence_len

        self.frame_path_list = sorted(sequence_frame_list, key = self.get_image_number)
        self.frame_path_list = [i for i in self.frame_path_list if self.get_image_number(i) in seg_mask_indices]
        assert len(self.frame_path_list) == self.sequence_len
        
    @staticmethod
    def factor_difficulty_check(factor, difficulty):
        assert factor in ['clutter', 'illumination', 'occlusion', 'pixel']
        if factor == 'clutter':
            assert difficulty != 2, "Annotations in highly cluttered scene are irregular. So not supported."
            return ['Low', 'Normal', 'High'][difficulty]
        elif factor == 'illumination':
            return ['Strong', 'Normal', 'Weak'][difficulty]
        elif factor == 'occlusion':
            return ['0%', '25%', '50%'][difficulty]
        elif factor =='pixel':
            return ['200', '30-200', '30'][difficulty]
        else:
            raise ValueError(f"Got unexpected factor {factor} and difficulty {difficulty}")
    
    @staticmethod
    def search_mask_folder(root, factor_name, difficulty_name, object_name):
        for i in range(1, 8):
            potential_base_dir = os.path.join(root, 'mask_and_bbox', f'scence#{i}', factor_name, difficulty_name, object_name)
            if os.path.exists(potential_base_dir):
                return potential_base_dir, i
        raise ValueError(f"Unable to find factor {factor_name} for object {object_name} with diff {difficulty_name}")
    
    def get_image_number(self, path_or_fn, fn_suffix='.jpg'):
        '''
        Get integer number of raw RGB JPG or binary mask PNG.
        '''
        raw_fn = os.path.basename(path_or_fn)
        assert self.scene_num is not None
        assert 'show' not in raw_fn # avoid visualized sample from mixing in
        if self.scene_num < 6:
            re1 = re.compile(r'frame\d+{}'.format(fn_suffix)) # scene 1,2,3,4,5
            assert re1.match(raw_fn)
        elif self.scene_num < 8:
            re2 = re.compile(r'color-\d+{}'.format(fn_suffix)) # scene 6,7
            assert re2.match(raw_fn)
        else:
            raise NotImplementedError
        num_re = re.compile(r'\d+')
        return int(num_re.search(raw_fn).group(0))
    
    def __getitem__(self, idx):
        assert idx < self.sequence_len and idx >= 0
        img_path = self.frame_path_list[idx]
        mask_path = self.seg_mask_path_list[idx]
        assert self.get_image_number(img_path) == self.get_image_number(mask_path, fn_suffix='.png')
        img_np = np.array(Image.open(img_path).convert('RGB'))
        mask_np = np.array(Image.open(mask_path)) # 0 and 255
        mask_np[mask_np == 255] = 1
        return img_np, mask_np
    
    def __len__(self):
        return self.sequence_len
