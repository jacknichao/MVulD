# -*- coding: utf-8 -*-
from __future__ import print_function, division
import sys
sys.path.insert(0,'.')
import os
import dgl
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import pdb
import torch as th

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv, GraphConv,GatedGraphConv

# Custom fusion modules
from .fusion import *
from .Rs_GCN import *
from .swin_transformer_v2 import *
from .build import *
from utils import load_checkpoint,auto_resume_helper, \
    reduce_tensor,resume_bestf1_helper,save_bestf1_checkpoint



def unbatch_features(g,max_node):
    h_i = []
    hf_i = []
    unix_i = [] 
    for g_i in dgl.unbatch(g): 
        h_i.append(g_i.ndata['HGATOUTPUT']) # (node num,512)
        hf_i.append(g_i.ndata['HFGATOUTPUT']) # (node num,4)
        unix_i.append(g_i.ndata["_UNIX_NODE_EMB"]) # (node num,768)
    max_len = max_node    
    for i, (v, k,u) in enumerate(zip(h_i, hf_i,unix_i)):
        if v.size(0)< max_len:
            h_i[i] = torch.cat(
            (v, torch.zeros(size=(max_len - v.size(0), *(v.shape[1:])), requires_grad=v.requires_grad,
                            device=v.device)), dim=0) 
            hf_i[i] = torch.cat(
            (k, torch.zeros(size=(max_len - k.size(0), *(k.shape[1:])), requires_grad=k.requires_grad,
                            device=k.device)), dim=0)
            unix_i[i] = torch.cat(
            (u, torch.zeros(size=(max_len - u.size(0), *(u.shape[1:])), requires_grad=k.requires_grad,
                            device=u.device)), dim=0)
        else:
            h_i[i] = h_i[i][0:max_len]
            hf_i[i] = hf_i[i][0:max_len]      
            unix_i[i] = unix_i[i][0:max_len]               
    return h_i,hf_i,unix_i

def pre_unbatch_features(g):
    h_i = []
    hf_i = []
    max_len = -1
    for g_i in dgl.unbatch(g): 
        h_i.append(g_i.ndata['HGATOUTPUT']) 
        hf_i.append(g_i.ndata['HFGATOUTPUT']) 
        max_len = max(g_i.number_of_nodes(), max_len)
    for i, (v, k) in enumerate(zip(h_i, hf_i)):
        h_i[i] = torch.cat(
            (v, torch.zeros(size=(max_len - v.size(0), *(v.shape[1:])), requires_grad=v.requires_grad,
                            device=v.device)), dim=0) 
        hf_i[i] = torch.cat(
            (k, torch.zeros(size=(max_len - k.size(0), *(k.shape[1:])), requires_grad=k.requires_grad,
                            device=k.device)), dim=0)                    
    return h_i,hf_i

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


class Multi_DefectModel_noGlobalImage(nn.Module):
    def __init__(self, config, pretrained=True, attention=True):
        super(Multi_DefectModel_noGlobalImage, self).__init__()

        self.num_features = 1024 # self.swinV2.output_num()
        self.config = config
        self.num_classes = config.MODEL.NUM_CLASSES
       
        hfeat: int = 512
        embfeat : int = 768 
        numheads = 4 # num_heads：Number of heads in Multi-Head Attention
        gatdrop: float = 0.2 # feat_drop：Dropout rate on feature
        mlpdropout: float = 0.2
        hdropout: float = 0.2

        gnn_args = {"out_feats": hfeat} # Output feature size
        gnn = GATConv
        gat_args = {"num_heads": numheads, "feat_drop": gatdrop} 
        gnn1_args = {**gnn_args, **gat_args, "in_feats": embfeat}  # embfeat：Input feature size
        gnn2_args = {**gnn_args, **gat_args, "in_feats": hfeat * numheads} #hfeat: int = 512,numheads=4    
        ## model：gat2layer:
        self.gat = gnn(**gnn1_args)
        self.gat2 = gnn(**gnn2_args)
        fcin = hfeat * numheads  
        self.fc = th.nn.Linear(fcin, hfeat)
        self.fconly = th.nn.Linear(embfeat, hfeat)
        self.mlpdropout = th.nn.Dropout(mlpdropout)

        ## Hidden Layers
        self.fch = []
        for _ in range(8): 
            self.fch.append(th.nn.Linear(hfeat, hfeat))
        self.hidden = th.nn.ModuleList(self.fch)
        self.hdropout = th.nn.Dropout(hdropout)

        # GCN reasoning 
        gcn_dim_size = 512 
        self.Rs_GCN_1 = Rs_GCN(in_channels=gcn_dim_size, inter_channels=gcn_dim_size)
        self.Rs_GCN_2 = Rs_GCN(in_channels=gcn_dim_size, inter_channels=gcn_dim_size)
        self.Rs_GCN_3 = Rs_GCN(in_channels=gcn_dim_size, inter_channels=gcn_dim_size)
        self.Rs_GCN_4 = Rs_GCN(in_channels=gcn_dim_size, inter_channels=gcn_dim_size)
        self.Rs_GCN_5 = Rs_GCN(in_channels=gcn_dim_size, inter_channels=gcn_dim_size)
        self.Rs_GCN_6 = Rs_GCN(in_channels=gcn_dim_size, inter_channels=gcn_dim_size)
        self.Rs_GCN_7 = Rs_GCN(in_channels=gcn_dim_size, inter_channels=gcn_dim_size)
        self.Rs_GCN_8 = Rs_GCN(in_channels=gcn_dim_size, inter_channels=gcn_dim_size)

        ## text embedding
        self.bn_text = nn.BatchNorm1d(embfeat)
        self.ln_text = nn.LayerNorm(embfeat) 
        self.fc_text = th.nn.Linear(embfeat, hfeat)
        
        self.max_node = 100
        self.bn_gat = nn.BatchNorm1d(self.max_node) 
        self.fc_gat = th.nn.Linear(512, 480)
        self.bn_bbox = nn.BatchNorm1d(self.max_node) 
        self.fc_bbox = th.nn.Linear(4, 32)
        
        self.swinbn = nn.BatchNorm1d(self.num_features)
        self.swinfc = th.nn.Linear(self.num_features, hfeat)
       
        self.hbn = nn.BatchNorm1d(hfeat)
        self.hln = nn.LayerNorm(hfeat)
        self.hfc = th.nn.Linear(hfeat, hfeat)
        
        self.final_fc_bn = nn.BatchNorm1d(hfeat) 
        self.final_fc = th.nn.Linear(hfeat, self.num_classes) 

    def forward(self, g, img_embedding,func_text_embedding):
        
        func_text_embedding = self.bn_text(func_text_embedding)
        func_text_embedding = F.elu(self.fc_text(func_text_embedding))

        g2 = g # h_func的shape： 
        h_func = g.ndata["_FUNC_EMB"] # shape N X 768 
        h = g.ndata["_UNIX_NODE_EMB"] # line_embedding: shape N X 768
        # h_all = g.ndata["_ALL_NODE_EMB"] # shape N X 800

        h = self.gat(g, h) # h=node embedding vectors at current layer
        h = h.view(-1, h.size(1) * h.size(2)) 
        h = self.gat2(g, h) 
        h = h.view(-1, h.size(1) * h.size(2)) # [N,2048]
        h = self.mlpdropout(F.elu(self.fc(h))) # shape：N X  512
        h_func = self.mlpdropout(F.elu(self.fconly(h_func))) # shape：N X 512
        
        ## Hidden layers
        for _, hlayer in enumerate(self.hidden):
            h = self.hdropout(F.elu(hlayer(h))) 
            h_func = self.hdropout(F.elu(hlayer(h_func))) # N X 512

        bboxes_feats = g.ndata['pos_emb']  # [N,4]
        g.ndata['HGATOUTPUT'] = h 
        g.ndata['HFGATOUTPUT'] = bboxes_feats 
        h_i,pos_i,unix_i=unbatch_features(g,self.max_node) 
        h_i = torch.stack(h_i)  # BS X MAX NODE X 512 (bsx100x512)
        pos_i = torch.stack(pos_i) # BS X MAX NODE X 4

        h_i = F.elu(self.fc_gat(self.bn_gat(h_i))) # [bs,max node,480]
        pos_i = F.elu(self.fc_bbox(self.bn_bbox(pos_i))) # [bs,max node,32]

        GCN_img_emd = torch.cat((h_i, pos_i), dim=2)  # [bs,max node,512]
        
        GCN_img_emd = GCN_img_emd.permute(0, 2, 1) 
        GCN_img_emd, __ = self.Rs_GCN_1(GCN_img_emd)
        GCN_img_emd, __ = self.Rs_GCN_2(GCN_img_emd)
        GCN_img_emd, __ = self.Rs_GCN_3(GCN_img_emd)
        GCN_img_emd, __ = self.Rs_GCN_4(GCN_img_emd)
        GCN_img_emd, __ = self.Rs_GCN_5(GCN_img_emd)
        GCN_img_emd, __ = self.Rs_GCN_6(GCN_img_emd)
        GCN_img_emd, __ = self.Rs_GCN_7(GCN_img_emd)
        GCN_img_emd, affinity_matrix = self.Rs_GCN_8(GCN_img_emd)
        # -> B,N,D
        GCN_img_emd = GCN_img_emd.permute(0, 2, 1)
        GCN_img_emd = l2norm(GCN_img_emd) 
        GCN_img_emd = torch.mean(GCN_img_emd, dim=1) # 【bs，512】
        h_feature = GCN_img_emd

        # all_feats = torch.cat((h_feature,func_text_embedding), dim=1)
        all_feats = func_text_embedding * GCN_img_emd
        all_feats = self.final_fc(self.final_fc_bn(all_feats)) 
        return all_feats

class Multi_DefectModel_noFunc(nn.Module):
    def __init__(self, config, pretrained=True, attention=True):
        super(Multi_DefectModel_noFunc, self).__init__()

        self.num_features = 1024 # self.swinV2.output_num()
        self.config = config
        self.num_classes = config.MODEL.NUM_CLASSES
        
        hfeat: int = 512
        embfeat : int = 768 
        numheads = 4 # num_heads：Number of heads in Multi-Head Attention
        gatdrop: float = 0.2 # feat_drop：Dropout rate on feature
        mlpdropout: float = 0.2
        hdropout: float = 0.2

        gnn_args = {"out_feats": hfeat} # Output feature size
        
        gnn = GATConv
        gat_args = {"num_heads": numheads, "feat_drop": gatdrop} 
        gnn1_args = {**gnn_args, **gat_args, "in_feats": embfeat}  # embfeat：Input feature size
        gnn2_args = {**gnn_args, **gat_args, "in_feats": hfeat * numheads} #hfeat: int = 512,numheads=4    
        ## model：gat2layer:
        self.gat = gnn(**gnn1_args)
        self.gat2 = gnn(**gnn2_args)
        fcin = hfeat * numheads  
        self.fc = th.nn.Linear(fcin, hfeat) 
        self.fconly = th.nn.Linear(embfeat, hfeat)
        self.mlpdropout = th.nn.Dropout(mlpdropout)

        ## Hidden Layers
        self.fch = []
        for _ in range(8): 
            self.fch.append(th.nn.Linear(hfeat, hfeat))
        self.hidden = th.nn.ModuleList(self.fch)
        self.hdropout = th.nn.Dropout(hdropout)

        # GCN reasoning 
        gcn_dim_size = 512 
        self.Rs_GCN_1 = Rs_GCN(in_channels=gcn_dim_size, inter_channels=gcn_dim_size)
        self.Rs_GCN_2 = Rs_GCN(in_channels=gcn_dim_size, inter_channels=gcn_dim_size)
        self.Rs_GCN_3 = Rs_GCN(in_channels=gcn_dim_size, inter_channels=gcn_dim_size)
        self.Rs_GCN_4 = Rs_GCN(in_channels=gcn_dim_size, inter_channels=gcn_dim_size)
        self.Rs_GCN_5 = Rs_GCN(in_channels=gcn_dim_size, inter_channels=gcn_dim_size)
        self.Rs_GCN_6 = Rs_GCN(in_channels=gcn_dim_size, inter_channels=gcn_dim_size)
        self.Rs_GCN_7 = Rs_GCN(in_channels=gcn_dim_size, inter_channels=gcn_dim_size)
        self.Rs_GCN_8 = Rs_GCN(in_channels=gcn_dim_size, inter_channels=gcn_dim_size)

        self.bn_text = nn.BatchNorm1d(embfeat)
        self.ln_text = nn.LayerNorm(embfeat) 
        self.fc_text = th.nn.Linear(embfeat, hfeat)
    
        self.max_node = 100
        self.bn_gat = nn.BatchNorm1d(self.max_node) 
        self.fc_gat = th.nn.Linear(512, 480)
        self.bn_bbox = nn.BatchNorm1d(self.max_node) 
        self.fc_bbox = th.nn.Linear(4, 32)
        
        self.swinbn = nn.BatchNorm1d(self.num_features)
        self.swinfc = th.nn.Linear(self.num_features, hfeat)
        
        self.hbn = nn.BatchNorm1d(hfeat)
        self.hln = nn.LayerNorm(hfeat)
        self.hfc = th.nn.Linear(hfeat, hfeat)
    
        self.final_fc_bn = nn.BatchNorm1d(hfeat*2) 
        self.final_fc = th.nn.Linear(hfeat*2, self.num_classes) 

    def forward(self, g, img_embedding,func_text_embedding):
        
        ## 1.img feature shape: BS X 1024 ---> BS X 521
        x = self.swinfc(self.swinbn(img_embedding)) 
        x = F.elu(x) # nn.GELU x = F.gelu(x) 

        ## 2.func_text_embedding
        # func_text_embedding = self.bn_text(func_text_embedding)
        # func_text_embedding = F.elu(self.fc_text(func_text_embedding))

        
        g2 = g # h_func的shape： N x 768
        h_func = g.ndata["_FUNC_EMB"] # shape N X 768 
        h = g.ndata["_UNIX_NODE_EMB"] # line_embedding: shape N X 768
       
        h = self.gat(g, h) # h=node embedding vectors at current layer
        h = h.view(-1, h.size(1) * h.size(2)) 
        h = self.gat2(g, h) 
        h = h.view(-1, h.size(1) * h.size(2)) # [N,2048]
        h = self.mlpdropout(F.elu(self.fc(h))) # shape：N X  512
        h_func = self.mlpdropout(F.elu(self.fconly(h_func))) # shape：N X 512
        
        ## Hidden layers
        for _, hlayer in enumerate(self.hidden):
            h = self.hdropout(F.elu(hlayer(h))) 
            h_func = self.hdropout(F.elu(hlayer(h_func))) # N X 512

        bboxes_feats = g.ndata['pos_emb']  # [N,4]
        g.ndata['HGATOUTPUT'] = h 
        g.ndata['HFGATOUTPUT'] = bboxes_feats 
        h_i,hf_i,unix_i=unbatch_features(g,self.max_node) # SHAPE: BS X MAX NODE X hfeat
        h_i = torch.stack(h_i)  # BS X MAX NODE X 512 (bsx100x512)
        pos_i = torch.stack(hf_i) # BS X MAX NODE X 4

        h_i = F.elu(self.fc_gat(self.bn_gat(h_i))) # [bs,max node,480]
        pos_i = F.elu(self.fc_bbox(self.bn_bbox(pos_i))) # [bs,max node,32]

        GCN_img_emd = torch.cat((h_i, pos_i), dim=2)  # [bs,max node,512]
        GCN_img_emd = GCN_img_emd.permute(0, 2, 1) 
        GCN_img_emd, __ = self.Rs_GCN_1(GCN_img_emd)
        GCN_img_emd, __ = self.Rs_GCN_2(GCN_img_emd)
        GCN_img_emd, __ = self.Rs_GCN_3(GCN_img_emd)
        GCN_img_emd, __ = self.Rs_GCN_4(GCN_img_emd)
        GCN_img_emd, __ = self.Rs_GCN_5(GCN_img_emd)
        GCN_img_emd, __ = self.Rs_GCN_6(GCN_img_emd)
        GCN_img_emd, __ = self.Rs_GCN_7(GCN_img_emd)
        GCN_img_emd, affinity_matrix = self.Rs_GCN_8(GCN_img_emd)
        # -> B,N,D
        GCN_img_emd = GCN_img_emd.permute(0, 2, 1)
        GCN_img_emd = l2norm(GCN_img_emd) 
        # print(f"GCN_img_emd.shape={GCN_img_emd.shape}")
        # GCN_img_emd = self.gcn_bn(pre_node_emb + GCN_img_emd) 
        GCN_img_emd = torch.mean(GCN_img_emd, dim=1) # 【bs，512】
        h_feature = GCN_img_emd

        all_feats = torch.cat((x,h_feature), dim=1) 
        all_feats = self.final_fc(self.final_fc_bn(all_feats)) 
        return all_feats