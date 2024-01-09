# from __future__ import print_function, division

import torch
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from sklearn.preprocessing import StandardScaler
import random
from dgl.data.utils import load_graphs, save_graphs
import torch.utils.data as data
import os
import os.path
import pandas as pd
import torch as th
import dgl
from tqdm import tqdm
from glob import glob
import sastvd as svd
import sastvd.helpers.joern as svdj



def make_dataset(image_list, labels):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        else:
            images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images

def pil_loader(path):
    with open(path, 'rb') as f: 
        with Image.open(f) as img:
            return img.convert('RGB')
            # return Image.open(path).convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    # from torchvision import get_image_backend
    # if get_image_backend() == 'accimage':
    #    return accimage_loader(path)
    # else:
    return pil_loader(path)


class BigVulImageList(object):
    def __init__(self, image_list, gtype="all",labels=None, transform=None, target_transform=None,
                 loader=default_loader):
        imgs = make_dataset(image_list, labels)
        # print(imgs number = len(imgs))
        self.imgs = imgs
        self.graph_type = gtype 
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index] 
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img) 
        if self.target_transform is not None:
            target = self.target_transform(target)
        _id = path.split('/')[-1].rstrip('.png')
        return img,target

    def __len__(self):
        return len(self.imgs)

    def cache_swin_features(self, swin=None):
        """Cache imgs features using swinv2 model.
        ONLY NEEDS TO BE RUN ONCE.:
        """
        savedir = svd.get_dir(svd.cache_dir() / "swinv2_method_level")
        done = [int(i.split("/")[-1].split(".")[0]) for i in glob(str(savedir / "*"))]
        done = set(done)
        print(savedir)
        batch_size = 4 
        batches = svd.chunks((range(len(self.imgs))), batch_size) 
        print(batches)
        for idx_batch in tqdm(batches):
            # print(imgs[idx_batch[0] : idx_batch[-1] + 1])
            temp= self.imgs[idx_batch[0] : idx_batch[-1] + 1] 
            batch_path = [i[0] for i in temp] 
            batch_target = [i[1] for i in temp] 
            # _id : str to int
            batch_ids = [int(path.split('/')[-1].rstrip('.png')) for path in batch_path]
            batch_imgs = [self.transform(self.loader(path)).tolist() for path in batch_path]
            batch_imgs= th.tensor(batch_imgs)
            print(batch_imgs.shape) # batch_imgs type:list batch_imgs[0] type:tensor
            if set(batch_ids).issubset(done):
                # print("swin features already exist")
                continue
            img_feature=swin.forward_features(batch_imgs).detach().cpu()
            print(img_feature.shape) # batch_size x 1024
            assert len(batch_imgs) == len(batch_ids)
            for i in range(len(batch_imgs)):
                th.save(img_feature[i], savedir / f"{batch_ids[i]}.pt")
                # print(batch_ids[i])
                # print(img_feature[i]) # shape:[1,1024]
                # print(savedir / f"{batch_ids[i]}.pt")
                 
