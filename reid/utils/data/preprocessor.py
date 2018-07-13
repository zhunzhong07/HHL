from __future__ import absolute_import
import os.path as osp

from PIL import Image
import torch
import numpy as np


class Preprocessor(object):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, fname, pid, camid


class CameraPreprocessor(object):
    def __init__(self, dataset, root=None, target_path=None, target_camstyle_path=None, transform=None, num_cam=6):
        super(CameraPreprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.target_path = target_path
        self.target_camstyle_path = target_camstyle_path
        self.transform = transform
        self.num_cam = num_cam

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        img_all = []
        fname_all = []
        pid_all = []
        camid_all = []
        fname, _, camid = self.dataset[index]
        # randomly assign pseudo label to unlabeled target image
        pid = int(torch.rand(1) * 10000 + 1000)
        if self.root is not None:
            for i in range(self.num_cam+1):
                if i == 0:
                    fpath = osp.join(self.root, self.target_path, fname)
                else:
                    fpath = osp.join(self.root, self.target_camstyle_path,
                                     fname[:-4] + '_fake_' + str(i) + '.jpg')
                img = Image.open(fpath).convert('RGB')
                if self.transform is not None:
                    img = self.transform(img)
                img_all.append(img)
                fname_all.append(fname)
                pid_all.append(pid)
                camid_all.append(camid)

        return img_all, fname_all, pid_all, camid_all
