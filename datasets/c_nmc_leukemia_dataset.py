from glob import glob
import pandas as pd

from datasets.dataset import Dataset


class CNmcLeukemiaTrainingDataset(Dataset):

  def __init__(self, path, transform=None):
    super().__init__(path, transform)

  
  def _get_imgs_paths(self):
    imgs_paths = sorted(glob(self.path + '/*/*.bmp'))
    imgs_lbls = [path.split('/')[-2] for path in imgs_paths]

    return imgs_paths, imgs_lbls


class CNmcLeukemiaTestingDataset(Dataset):

  def __init__(self, path, transform=None):
    super().__init__(path, transform)

  
  def _get_imgs_paths(self):
    imgs_paths = sorted(glob(self.path + '/*.bmp'))
    csv_path = glob(self.path + '/*.csv')[0]

    df = pd.read_csv(csv_path)
    imgs_nms = df['new_names'].values
    lbls = df['labels'].apply(lambda lbl: 'hem' if lbl == 0 else 'all').values
    imgs_nm_to_lbl = {img_nm: lbl for img_nm, lbl in zip(imgs_nms, lbls)}

    imgs_lbls = [imgs_nm_to_lbl[path.split('/')[-1]] for path in imgs_paths]

    return imgs_paths, imgs_lbls
