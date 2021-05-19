
# Data set class to load
# LR HR Image Transaforms 
# Trains Test folder things


from torch.utils.data import Dataset
import torch 
from torchvision import transforms
from PIL import Image
import os
from glob import glob 

class SrGanDataset(Dataset):

    def __init__(self, dir_, std, mean, hr_shape):
        super(SrGanDataset, self).__init__()
        self.mean = mean
        self.std = std
        self.hr_height, self.hr_width = hr_shape
        self.files = glob(dir_+ "/*")

        self.lr_transforms = transforms.Compose([
            transforms.Resize((self.hr_height//4,  self.hr_width//4), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean = self.mean, std = self.std)
            
        ])
        
        self.hr_transforms = transforms.Compose([
            transforms.Resize((self.hr_height,  self.hr_width), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean = self.mean, std = self.std)
            
        ])


    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        img = Image.open(self.files[index])

        img_lr = self.lr_transforms(img)
        img_hr = self.hr_transforms(img)

        return img_lr, img_hr




