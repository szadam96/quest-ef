import os
import random

import skimage
from skimage import transform
import cv2
from sklearn.model_selection import GroupShuffleSplit
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import json
from torchvision import transforms
import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import minmax_scale
import pandas as pd
from pathlib import Path

class EchoImageDataset(Dataset):
    """Dataset class for echocardigrapic images"""
    
    def __init__(self, root_dir, transform, data_csv=None, mask_file_name='mask.png', include_binary_mask=False):
        root_dir = Path(root_dir)
        if not data_csv is None:
            data_df = pd.read_csv(data_csv)
            self.data = [dicom_id for dicom_id in data_df['dicom_id'].values if os.path.exists(root_dir / dicom_id)]
        else:
            self.data = [dicom_id for dicom_id in os.listdir(root_dir)]

        self.root_dir = root_dir
        self.transform = transform
        self.mask_file_name = mask_file_name
        self.frame_list = self.create_frame_list()
        self.include_binary_mask = include_binary_mask
        
    def create_frame_list(self):
        key_list = []
        self.id_list = []
        for dicom_id in self.data:
            frames = os.listdir(self.root_dir / dicom_id / 'frames')
            for frame_name in frames:
                img_path = self.root_dir / dicom_id / 'frames' / frame_name
                mask_path = self.root_dir / dicom_id / self.mask_file_name
                key_list.append((img_path, mask_path))
                self.id_list.append(dicom_id)
                    
        return key_list

    def to_tensor(self, img, mask):
        expanded_frame = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if expanded_frame.dtype == np.uint8:
            expanded_frame = expanded_frame.astype(np.float32)
            expanded_frame /= 255.0
        if self.include_binary_mask:
            expanded_frame[:, :, 0] = mask

        frame_tensor = ToTensorV2()(image=expanded_frame)['image']

        return frame_tensor
        
    def __len__(self):
        return len(self.frame_list)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path, mask_path = self.frame_list[idx]

        dcm_id = self.id_list[idx]
        
        img = np.asarray([cv2.imread(str(img_path))[:,:,0]])
        img = np.transpose(img, (1, 2, 0))
        binary_mask = (cv2.imread(str(mask_path))/255.0)[:,:,0]
        
        
        if self.transform:
            transformed = self.transform(image=img, mask=binary_mask)
            img = transformed['image']
            binary_mask = transformed['mask']

        input_tensor = self.to_tensor(img, binary_mask)

        sample = {'input_tensor': input_tensor, 'dicom_id': dcm_id}
 
            
        return sample
