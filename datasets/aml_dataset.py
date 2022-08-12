from glob import glob
from datasets.dataset import Dataset

EXCLUDE_LBLS = {'KSC', 'LYA', 'MMZ', 'MOB'}

class AMLDataset(Dataset):

  def __init__(self, path, transform=None):
    super().__init__(path, transform, EXCLUDE_LBLS)
  
  def _get_imgs_paths(self):
    imgs_paths = sorted(glob(self.path + '/*/*.tiff'))
    imgs_lbls = [path.split('/')[-1].split('_')[0] for path in imgs_paths]

    return imgs_paths, imgs_lbls
