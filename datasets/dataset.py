import torch
import torchvision
import numpy as np
from PIL import Image
from glob import glob
from abc import ABC, abstractmethod


class Dataset(torch.utils.data.Dataset, ABC):

  def __init__(self, path, transform=None, exclude_lbls=None):
    self.path = path
    self.transform = transform

    if exclude_lbls is None:
        self.exclude_lbls = set()
    else:
        self.exclude_lbls = exclude_lbls
        
    self.imgs_paths, self.imgs_lbls = None, None
    self.lbl_encoder, self.lbl_encoder_inverse = None, None
    self.num_classes, self.lbls_sorted = None, None
    self._get_imgs_list()

  @abstractmethod
  def _get_imgs_paths(self):
    # returns imgs_paths and imgs_lbls
    pass

  def _get_imgs_list(self):
    imgs_paths, imgs_lbls = self._get_imgs_paths()
    
    self.imgs_paths, self.imgs_lbls = [], []
    for path, lbl in zip(imgs_paths, imgs_lbls):
        if lbl not in self.exclude_lbls:
            self.imgs_paths.append(path)
            self.imgs_lbls.append(lbl)
        
    self.lbls_sorted = sorted(list(set(self.imgs_lbls)))
    self.num_classes = len(self.lbls_sorted)
    
    self.lbl_encoder = {lbl: idx for idx, lbl in enumerate(self.lbls_sorted)}
    self.lbl_encoder_inverse = {idx: lbl for idx, lbl in enumerate(self.lbls_sorted)}

    self.imgs_lbls = [self.lbl_encoder[lbl] for lbl in self.imgs_lbls]

  def __len__(self):
    return len(self.imgs_paths)

  def __getitem__(self, idx):
    
    path = self.imgs_paths[idx]
    lbl = self.imgs_lbls[idx]

    im = Image.open(path) 
    im = im.resize((224, 224))   # for models built for imagenet
    im = np.array(im)
    im = im[:, :, :3]   # remove the 4th channel

    if self.transform:
      im = self.transform(im)

    return torchvision.transforms.ToTensor()(im), lbl