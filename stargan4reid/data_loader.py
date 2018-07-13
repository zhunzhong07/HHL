from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import os.path as osp
from glob import glob
import re


class ReidDataset(data.Dataset):
    def __init__(self, image_path, transform):
        self.image_path = image_path
        self.transform = transform
        self.fnames = []
        self.cams = []
        self.preprocess()
        self.num_data = int(len(self.fnames))

    def preprocess(self):

        pattern = re.compile(r'([-\d]+)_c(\d)')
        fpaths = sorted(glob(osp.join(self.image_path, '*.jpg')))
        for fpath in fpaths:
            fname = osp.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
            if pid == -1: continue  # junk images are just ignored
            cam -= 1
            self.fnames.append(fpath)
            self.cams.append(cam)

    def __getitem__(self, index):
        image = Image.open(self.fnames[index])
        cam = self.cams[index]
        name = osp.basename(self.fnames[index])
        return self.transform(image), cam, name

    def __len__(self):
        return self.num_data


def get_loader(image_path, batch_size=16, mode='train', num_workers=4):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    if mode == 'train':
        transform.append(T.Resize((288, 144)))
        transform.append(T.RandomCrop((256, 128)))
    else:
        transform.append(T.Resize((256, 128)))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dataset = ReidDataset(image_path, transform)

    data_loader = data.DataLoader(dataset=dataset,
                             batch_size=batch_size, num_workers=num_workers,
                             shuffle=(mode == 'train'), pin_memory=True, drop_last=(mode == 'train'))
    return data_loader