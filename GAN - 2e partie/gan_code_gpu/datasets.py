# Configure the dataloader

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torchvision import datasets
import glob
import os
from pathlib import Path
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)

        self.files_img = sorted(glob.glob(os.path.join(root, mode) + '/*.jpg'))
        if mode == 'val':
            self.files_img.extend(
                sorted(glob.glob(os.path.join(root, 'val') + '/*.jpg')))

        self.files_mask = sorted(glob.glob(os.path.join(root, mode) + '/*.png'))
        if mode == 'val':
            self.files_mask.extend(
                sorted(glob.glob(os.path.join(root, 'val') + '/*.png')))
            
        assert len(self.files_img) == len(self.files_mask)

    def __getitem__(self, index):

        img = Image.open(self.files_img[index % len(self.files_img)])
        mask = Image.open(self.files_mask[index % len(self.files_img)])
        mask = mask.convert('RGB')

        img = self.transform(img)
        mask = self.transform(mask)

        return img, mask

    def __len__(self):
        return len(self.files_img)