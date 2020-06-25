import re
import random
from operator import itemgetter
from pathlib import Path
from itertools import repeat
from functools import partial
from typing import Any, Callable, BinaryIO, Dict, List, Match, Pattern, Tuple, Union, Optional

import torch
import glob
import numpy as np
from torch import Tensor
from PIL import Image
from torchvision import transforms
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader
from utilities import class2one_hot



#the args.dataset folder needs to contain TRAIN, VAL folders. Each of them should contain in_npy and gt_npy folders, containing input channels and ground truths. 
#the working of loaders depends on the network used. If UNet, it doesnt matter so much (though input imgs shouldnt be too small). For deep medic, The whole image X is
#subsampled by factor three to be input to the second pathway, and the size of the input to the first pathway is then X-30. But X should be =-2 (mod 3). So padding with 
#zeros is done first to get this requirements satisfied. 

def get_loaders(netname, dataset, batch_size, n_class, debug, usingBL):
    #check if appropriate folder structure exists:
    assert (Path(dataset, "VAL/in_npy").exists() and Path(dataset, "TRAIN/in_npy").exists()), f"Input data missing or folder structure wrong in {dataset}."
    assert (Path(dataset, "VAL/gt_npy").exists() and Path(dataset, "TRAIN/gt_npy").exists()), f"Ground truth missing or folder structure wrong in {dataset}."
    assert not usingBL or (Path(dataset, "VAL/dt_npy").exists() and Path(dataset, "TRAIN/dt_npy").exists()), f"Distance data missing or folder structure wrong in {dataset}."
    #check also that we have the same amount of data in all folders:

    groundtruth = transforms.Compose([
        lambda img: np.array(img)[np.newaxis, ...],
        lambda nd: torch.tensor(nd, dtype=torch.int64),
        partial(class2one_hot, C=n_class),
        itemgetter(0)
    ])

    origpath = transforms.Compose([
        #lambda im: np.array(im)[np.newaxis, ...],
        lambda img: np.pad(img,[(0,0)]+[((img.shape[i]+2)%3, (img.shape[i]+2)%3) for i in range(1,len(img.shape))], mode='constant', constant_values=0),
        lambda im: im[:, 15:-15, 15:-15], #here we assume imgs ar big enough for this. and they are 2D; channels x height x width 
        lambda nd: torch.tensor(nd, dtype=torch.float32),
        lambda norm: (norm-torch.min(norm))/(torch.max(torch.abs(norm))+1e-10) #normalize
    ])

    subsampledpath = transforms.Compose([
        #lambda im: np.array(im)[np.newaxis, ...],
        lambda img: np.pad(img, [(0,0)]+[((img.shape[i]+2)%3, (img.shape[i]+2)%3) for i in range(1,len(img.shape))], mode='constant', constant_values=0), #pad
        lambda nd: torch.tensor(nd[tuple([slice(nd.shape[0])] + [slice(0, nd.shape[i], 3) for i in range(1,len(nd.shape))])], dtype=torch.float32), #subsample
        lambda norm: (norm-torch.min(norm))/(torch.max(torch.abs(norm))+1e-10) #normalize
    ])

    identity = transforms.Compose([
        lambda nd: torch.tensor(nd, dtype=torch.float32),
        lambda norm: (norm-torch.min(norm))/(torch.max(torch.abs(norm))+1e-10) #normalize
    ])
    
    distancetrans = transforms.Compose([
        #lambda img: np.array(img)[np.newaxis, ...],
        lambda nd: torch.tensor(nd, dtype=torch.float64)
    ])

    const = [groundtruth, distancetrans]
    if netname=="DeepMedic":
        transformi = [origpath, subsampledpath] + const
    elif netname=="UNet":
        transformi = [identity] + const
    else:
        raise f"Network {netname} not implemented!"

    val_files = glob.glob(dataset+"/VAL/in_npy/*")
    val_gts = glob.glob(dataset+"/VAL/gt_npy/*")
    val_dts = glob.glob(dataset+"/VAL/dt_npy/*")
    val_dataset = MyDataset(val_files, val_gts, val_dts, transformi, n_class=n_class, debug=debug)
    val_loader = DataLoader(val_dataset, num_workers=batch_size + 5, pin_memory=True, 
                                batch_size=batch_size, shuffle=False, drop_last=True)

    train_files = glob.glob(dataset+"/TRAIN/in_npy/*")
    train_gts = glob.glob(dataset+"/TRAIN/gt_npy/*")
    train_dts = glob.glob(dataset+"/TRAIN/dt_npy/*")
    train_dataset = MyDataset(train_files, train_gts, train_dts, transformi, n_class=n_class, debug=debug)
    train_loader = DataLoader(train_dataset, num_workers=batch_size + 5, pin_memory=True, 
                                batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, val_loader


class MyDataset(Dataset):
    def __init__(self, filenames: List[str], gtnames: List[str], dtnames: List[str], transforms: List[Callable], 
                     n_class: int, debug=False, quiet=True) -> None:

        self.filenames = filenames; self.filenames.sort() #the data files are expected to have the same names and corresponding unique IDs!!
        self.gtnames = gtnames; self.gtnames.sort()
        self.dtnames = dtnames; self.dtnames.sort() #if dtnames empty, we assume we arent using dts. 
        if debug:
            self.filenames = self.filenames[:10]
            self.gtnames = self.gtnames[:10]
            self.dtnames = self.dtnames[:10]
    
        self.transforms = transforms #contains [n transforms for inp, 1 transform for gt, 1 transform for dt]


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index: int) -> List[Any]:
        inp = np.load(self.filenames[index])
        gt = self.transforms[-2](np.load(self.gtnames[index]))
        #get transformed inputs:
        transformed_in = [tr(inp) for tr in self.transforms[:-2]]
        if len(self.dtnames)!=0:
            return transformed_in, gt, self.transforms[-1](np.load(self.dtnames[index]))
        else:
            return transformed_in, gt