from __future__ import print_function, absolute_import
import os.path as osp
import numpy as np
import pdb
from glob import glob
import re


class DA(object):

    def __init__(self, data_dir, source, target):

        # source / target image root
        self.source_images_dir = osp.join(data_dir, source)
        self.target_images_dir = osp.join(data_dir, target)
        # training image dir
        self.source_train_path = 'bounding_box_train'
        self.target_train_path = 'bounding_box_train'
        self.target_train_camstyle_path = 'bounding_box_train_camstyle_stargan4reid'
        self.gallery_path = 'bounding_box_test'
        self.query_path = 'query'

        self.source_train, self.target_train, self.query, self.gallery = [], [], [], []
        self.num_train_ids, self.num_query_ids, self.num_gallery_ids = 0, 0, 0
        self.target_num_cam = 6 if 'market' in target else 8
        self.source_num_cam = 6 if 'market' in source else 8
        self.load()

    def preprocess(self, images_dir, path, relabel=True):
        pattern = re.compile(r'([-\d]+)_c(\d)')
        all_pids = {}
        ret = []
        fpaths = sorted(glob(osp.join(images_dir, path, '*.jpg')))
        for fpath in fpaths:
            fname = osp.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
            if pid == -1: continue
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            pid = all_pids[pid]
            cam -= 1
            ret.append((fname, pid, cam))
        return ret, int(len(all_pids))

    def load(self):
        self.source_train, self.num_train_ids = self.preprocess(self.source_images_dir, self.source_train_path)
        self.target_train, _ = self.preprocess(self.target_images_dir, self.target_train_path)
        self.gallery, self.num_gallery_ids = self.preprocess(self.target_images_dir, self.gallery_path, False)
        self.query, self.num_query_ids = self.preprocess(self.target_images_dir, self.query_path, False)

        print(self.__class__.__name__, "dataset loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  source train    | {:5d} | {:8d}"
              .format(self.num_train_ids, len(self.source_train)))
        print("  target train    | 'Unknown' | {:8d}"
              .format(len(self.target_train)))
        print("  query    | {:5d} | {:8d}"
              .format(self.num_query_ids, len(self.query)))
        print("  gallery  | {:5d} | {:8d}"
              .format(self.num_gallery_ids, len(self.gallery)))
