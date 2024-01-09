#!/usr/bin/python
# -*- coding: utf-8 -*-
from importlib.resources import path
import dgl
import torch
import torch.utils.data as data
from timm.data import Mixup
from torchvision import datasets, transforms
from dgl.dataloading import GraphDataLoader
import os
import pickle
import random
import sys
import json
from PIL import Image
from data.build import build_transform
from models import build_model
from models.unixcoder import UniXcoder,MyUniXcoder
from data.data_list import ImageList
from data.swin_dataset import BigVulImageList

import torch.distributed as dist
from utils import load_checkpoint,auto_resume_helper, \
    reduce_tensor,resume_bestf1_helper,save_bestf1_checkpoint

sys.path.insert(0, '.')

import numpy as np
from skimage import io

def load_swin_checkpoint(config,model): 
    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT) 
        if resume_file:
            if config.MODEL.RESUME: 
                print(f"BIGVUL: auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file 
            config.freeze()
            print(f'BIGVUL: auto resuming from {resume_file}') 
        else:
            print(f'BIGVUL: no checkpoint found in {config.OUTPUT}, ignoring auto resume')
            return 
        
    if config.MODEL.RESUME:
        print(f"======> BIGVUL:Resuming form {config.MODEL.RESUME}....................")
        if config.MODEL.RESUME.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                config.MODEL.RESUME, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
        if config.MODEL.NUM_CLASSES!=1000 and checkpoint['model']['head.weight'].shape[0] == 1000:
            checkpoint['model']['head.weight'] = torch.nn.Parameter(
                torch.nn.init.xavier_uniform_(torch.empty(config.MODEL.NUM_CLASSES, 1024))) # swinv2 base =1024
            checkpoint['model']['head.bias'] = torch.nn.Parameter(torch.randn(config.MODEL.NUM_CLASSES))
        model.load_state_dict(checkpoint['model'], strict=False)
        del checkpoint
    return    

def load_bestf1_swin(config,model):
    # find best-f1 model
    if config.TRAIN.BEST_RESUME: 
        if not os.path.exists(config.OUTPUT):
            os.makedirs(config.OUTPUT)
        resume_file = resume_bestf1_helper(config.OUTPUT) # load the best-f1 swinV2 model
        if resume_file:
            if config.MODEL.RESUME:
                print(f"BIGVUL:swin-best-f1 resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file 
            config.freeze()
            print(f'BIGVUL: swin-best-f1 resuming from {resume_file}') 
        else:
            print(f'BIGVUL: swi-best-f1 no checkpoint found in {config.MODEL.RESUME}, ignoring best resume')
    if config.MODEL.RESUME: # set to '' when pretrain
        # max_accuracy,epoch = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        del checkpoint
    return 

def getPath(): 
    balanced_path = "/data1/username/project/MMVD/baselines/storage/results/unixcoder/bigvul/checkpoint-best-f1/pytorch_model.bin"
    path = balanced_path
    return path

def build_MyUniXcoder():
    ## 1. build the initial unixcoder model 
    model_name_or_path = "microsoft/unixcoder-base-nine"
    unixcoder = UniXcoder(model_name_or_path)
    config = unixcoder.config
    model = unixcoder.model
    tokenizer = unixcoder.tokenizer
    tokenize = unixcoder.tokenize 
    myModel = MyUniXcoder(model, config, tokenizer,tokenize) 
    # 2. load the best-f1 unixcoder model
    path = getPath()
    myModel.load_state_dict(torch.load(path))
    return myModel

def bigvul_dataset(config):

    ##1. build swinV2 model
    SWINMODEL= build_model(config) 
    # load best-f1 swinV2 model
    load_bestf1_swin(config,SWINMODEL) 

    # 2. build the best-f1 unxicoder
    myUniX = build_MyUniXcoder()

    transform_train = build_transform(is_train=True, config=config) # used to transform trainning images
    transform_test = build_transform(is_train=False, config=config) # used to transform val/testing images
    if config.DATA.DATASET == 'imagenet':
        g_type="all" # graph type: "pdg"，"cpg",etc
        print(f"-----gtype={g_type}------")
        train_data = ImageList(open(config.TRAIN.DATA_PATH).readlines(),gtype=g_type,transform=transform_train)
        val_data = ImageList(open(config.VAL.DATA_PATH).readlines(),gtype=g_type,transform=transform_test)
        test_data = ImageList(open(config.TEST.DATA_PATH).readlines(),gtype=g_type,transform=transform_test)
        
        ## only need to run once. You can comment it out later. 
        # Store the code embeddings from swing and unixcoder in advance
        train_data.cache_swin_features(SWINMODEL)
        val_data.cache_swin_features(SWINMODEL)
        test_data.cache_swin_features(SWINMODEL)

        train_data.cache_g_items(myUniX) 
        val_data.cache_g_items(myUniX) 
        test_data.cache_g_items(myUniX) 

    return train_data,val_data,test_data

def swin_bigvul_dataset(config):
    transform_train = build_transform(is_train=True, config=config) 
    transform_test = build_transform(is_train=False, config=config) 
    if config.DATA.DATASET == 'imagenet':
        train_data = BigVulImageList(open(config.TRAIN.DATA_PATH).readlines(),gtype="all",transform=transform_train)
        val_data = BigVulImageList(open(config.VAL.DATA_PATH).readlines(),gtype="all",transform=transform_test)
        test_data = BigVulImageList(open(config.TEST.DATA_PATH).readlines(),gtype="all",transform=transform_test)
        # nb_classes = config.MODEL.NUM_CLASSES 
    return train_data,val_data,test_data

def node_dl(config,g, shuffle=False):
    """Return node dataloader."""
    nsampling_hops=2
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(nsampling_hops) 
    # self.nsampling_hops=1 : 1 layerGNN
    return dgl.dataloading.NodeDataLoader(
        g,
        g.nodes(),
        sampler,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=shuffle,
        drop_last=False,
        num_workers=config.DATA.NUM_WORKERS,
        )

def bigvul_loader_graph(config,nsampling=False): 
    config.defrost()
    train_data,val_data,test_data = bigvul_dataset(config)
    config.freeze()
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
            train_data, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    
    if config.TEST.SEQUENTIAL:
        sampler_val = torch.utils.data.SequentialSampler(val_data)
        sampler_test = torch.utils.data.SequentialSampler(test_data)
    else:
        sampler_val = torch.utils.data.distributed.DistributedSampler(
            val_data, shuffle=config.TEST.SHUFFLE
        )
        sampler_test = torch.utils.data.distributed.DistributedSampler(
            test_data, shuffle=config.TEST.SHUFFLE
        )

    data_loader_train = GraphDataLoader(
        train_data,
        sampler=sampler_train,
        shuffle=False, 
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True, 
    )

    data_loader_val = GraphDataLoader(
        val_data, 
        sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    data_loader_test = GraphDataLoader(
        test_data,
        sampler=sampler_test,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    # setup mixup / cutmix (config.AUG.MIXUP=0.8 
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)
    
    return train_data,val_data,test_data,data_loader_train,data_loader_val,data_loader_test,mixup_fn

def bigvul_loader_swin(config):
    config.defrost()
    train_data,val_data,test_data = swin_bigvul_dataset(config)
    config.freeze()
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
            train_data, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    
    if config.TEST.SEQUENTIAL:
        sampler_val = torch.utils.data.SequentialSampler(val_data)
        sampler_test = torch.utils.data.SequentialSampler(test_data)
    else:
        sampler_val = torch.utils.data.distributed.DistributedSampler(
            val_data, shuffle=config.TEST.SHUFFLE
        )
        sampler_test = torch.utils.data.distributed.DistributedSampler(
            test_data, shuffle=config.TEST.SHUFFLE
        )

   
    data_loader_train = torch.utils.data.DataLoader( 
        train_data, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        val_data, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    data_loader_test = torch.utils.data.DataLoader(
        test_data, sampler=sampler_test,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)
    
    return train_data,val_data,test_data,data_loader_train,data_loader_val,data_loader_test,mixup_fn
        

