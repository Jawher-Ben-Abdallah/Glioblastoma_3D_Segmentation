import numpy as np
import pickle
from skimage.filters import gaussian

import torch
from torch.utils.data import Dataset
from utils.utils import crop, pad_image


class BraTS20Dataset(Dataset):
    def __init__(self, patients, args, mode="train"):
        if not isinstance(args.patch_size, (list, np.ndarray)):
            self.patch_size = [args.patch_size] * 3
        else:
            self.patch_size = args.patch_size
            
        if not isinstance(args.val_size, (list, np.ndarray)):
            self.val_size = [args.val_size] * 3
        else:
            self.val_size = args.val_size
            
        self.modalities = 4
        self.mode = mode
        self.augment = args.augment
        self.patients = patients

    def random_augmentation(self, probability, augmented, original):
        condition = self.coin_flip(probability=probability)
        neg_condition = condition ^ True
        return condition * augmented + neg_condition * original

    def noise_fn(self, img, p):
        img_noised = img + np.random.normal(scale=np.random.uniform(high=0.33), size=img.shape)
        return self.random_augmentation(p, img_noised, img)

    def blur_fn(self, img, p):
        img_blurred = np.zeros_like(img)
        gaussian(img, output=img_blurred, sigma=np.random.uniform(low=0.5, high=1.5), channel_axis=0)
        return self.random_augmentation(p, img_blurred, img)

    def brightness_fn(self, img, p):
        brightness_scale = self.random_augmentation(p, np.random.uniform(low=0.7, high=1.3), 1.0)
        return brightness_scale * img

    def contrast_fn(self, img, p):
        scale = self.random_augmentation(p, np.random.uniform(low=0.65, high=1.3), 1.0)
        return np.clip(scale * img, img.min(), img.max(), out=np.zeros_like(img))

    def flips_fn(self, img, lbl):
        horizontal = self.coin_flip()
        vertical = self.coin_flip()
        depthwise = self.coin_flip()
        img, lbl = self.flip(img, lbl, 1) if depthwise == True else (img, lbl)
        img, lbl = self.flip(img, lbl, 2) if vertical == True else (img, lbl)
        img, lbl = self.flip(img, lbl, 3) if horizontal == True else (img, lbl)
        return img, lbl
    
    def __len__(self):
        return len(self.patients)
    
    def __getitem__(self, idx):
        patient = self.patients[idx]
        patient_id = patient.split('/')[-1]
        patient_data, patient_metadata = self.load_patient(patient)
        
        if self.mode == "train":
            patient_data = pad_image(patient_data, [i + 1 for i in self.patch_size])
            img, lbl = crop(patient_data[:-1][None], patient_data[-1:][None], self.patch_size[0])
            img, lbl = img[0], lbl[0]
            
            if self.augment:
                img, lbl = self.flips_fn(img, lbl)
                img = self.noise_fn(img, p=.2)
                img = self.blur_fn(img, p=.2)
                img = self.brightness_fn(img, p=.2)
                img = self.contrast_fn(img, p=.2)
        else:
            patient_data = pad_image(patient_data, self.val_size)
            img, lbl = patient_data[:-1], patient_data[-1:]
             
        img = torch.from_numpy(img.copy()).type(torch.float32)
        lbl = torch.from_numpy(lbl.copy()).type(torch.float32)
        return img, lbl, patient_id
         
    @staticmethod
    def load_patient(patient):
        data = np.load(patient +'.npy', mmap_mode='r')
        with open(patient + '.pkl', 'rb') as f:
            metadata = pickle.load(f)
        return data, metadata
    
    @staticmethod
    def coin_flip(probability=0.5):
        return np.random.choice([False, True], size=1, p=[1. - probability, probability]).item()
    
    @staticmethod
    def flip(img, lbl, axis):
        return np.flip(img, axis=axis), np.flip(lbl, axis=axis)