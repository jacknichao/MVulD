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
        h_i.append(g_i.ndata['HGATOUTPUT']) # node-level
        hf_i.append(g_i.ndata['HFGATOUTPUT']) # func-level
        max_len = max(g_i.number_of_nodes(), max_len)
    for i, (v, k) in enumerate(zip(h_i, hf_i)):
        h_i[i] = torch.cat(
            (v, torch.zeros(size=(max_len - v.size(0), *(v.shape[1:])), requires_grad=v.requires_grad,
                            device=v.device)), dim=0) # 
        hf_i[i] = torch.cat(
            (k, torch.zeros(size=(max_len - k.size(0), *(k.shape[1:])), requires_grad=k.requires_grad,
                            device=k.device)), dim=0)
    # shape：([batch_size,max_node_incurrentbatch, 768])                       
    return h_i,hf_i

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


## motivation: only Image feature
class Multi_DefectModel_Image(nn.Module):
    def __init__(self, config, pretrained=True, attention=True):
        super(Multi_DefectModel_Image, self).__init__()

        self.num_features = 1024 # self.swinV2.output_num()
        self.config = config
        self.num_classes = config.MODEL.NUM_CLASSES
        ## g_features
        hfeat: int = 512 # 
        embfeat : int = 768 # unix embedding dimension
        numheads = 4 # num_heads：Number of heads in Multi-Head Attention
        gatdrop: float = 0.2 # feat_drop：Dropout rate on feature
        mlpdropout: float = 0.2
        hdropout: float = 0.2
        ## swin:global img embedding
        self.swinbn = nn.BatchNorm1d(self.num_features)
        self.swinfc = th.nn.Linear(self.num_features, hfeat)
        ## classifier:
        self.final_fc = th.nn.Linear(self.num_features, self.num_classes) 
        self.final_fc_bn = nn.BatchNorm1d(self.num_features) 

    def forward(self, g, img_embedding,func_text_embedding): 
        # outputs = self.final_fc(self.final_fc_bn(img_embedding)) # bn
        outputs = self.final_fc(img_embedding) # 
        return outputs

## motivation: func-level code text
class Multi_DefectModel_FuncText(nn.Module):
    def __init__(self, config, pretrained=True, attention=True):
        super(Multi_DefectModel_FuncText, self).__init__()

        self.num_features = 1024 # self.swinV2.output_num()
        self.config = config
        self.num_classes = config.MODEL.NUM_CLASSES
        ## g_features
        hfeat: int = 512  
        embfeat : int = 768 
        numheads = 4 # num_heads：Number of heads in Multi-Head Attention
        gatdrop: float = 0.2 # feat_drop：Dropout rate on feature
        mlpdropout: float = 0.2
        hdropout: float = 0.2
        ## node features mlp:
        self.fconly = th.nn.Linear(embfeat, hfeat)
        self.mlpdropout = th.nn.Dropout(0.2)
        ## Hidden Layers
        self.fch = []
        for _ in range(8): 
            self.fch.append(th.nn.Linear(hfeat, hfeat))
        self.hidden = th.nn.ModuleList(self.fch)
        self.hdropout = th.nn.Dropout(hdropout)

        ## text embedding
        self.bn_text = nn.BatchNorm1d(embfeat)
        self.ln_text = nn.LayerNorm(embfeat) 
        self.fc_text = th.nn.Linear(embfeat, hfeat)
        ## classifier:
        self.final_fc = th.nn.Linear(embfeat, self.num_classes) 
        self.final_fc_bn = nn.BatchNorm1d(embfeat) 

    def forward(self, g, img_embedding,func_text_embedding):
        # outputs = self.final_fc(self.final_fc_bn(func_text_embedding))  
        outputs = self.final_fc(func_text_embedding)
        return outputs

## motivation: only graph
class Multi_DefectModel_Graph(nn.Module):
    def __init__(self, config, pretrained=True, attention=True):
        super(Multi_DefectModel_Graph, self).__init__()

        self.num_features = 1024 # self.swinV2.output_num()
        self.config = config
        self.num_classes = config.MODEL.NUM_CLASSES
        ## g_features
        hfeat: int = 512 
        embfeat : int = 768 
        numheads = 4 # num_heads：Number of heads in Multi-Head Attention
        gatdrop: float = 0.2 # feat_drop：Dropout rate on feature
        mlpdropout: float = 0.2
        hdropout: float = 0.2

        gnn_args = {"out_feats": hfeat} # Output feature size
        ## gat
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

        self.max_node = 100
        self.bn_gat = nn.BatchNorm1d(self.max_node) 
        self.fc_gat = th.nn.Linear(512, 480)
        self.bn_bbox = nn.BatchNorm1d(self.max_node) 
        self.fc_bbox = th.nn.Linear(4, 32)
        
        self.hbn = nn.BatchNorm1d(hfeat)
        self.hln = nn.LayerNorm(hfeat)
        self.hfc = th.nn.Linear(hfeat, hfeat)
        ## classifier:
        self.final_fc = th.nn.Linear(512, self.num_classes) 
        self.final_fc_bn = nn.BatchNorm1d(512) 

    def forward(self, g, img_embedding,func_text_embedding):

        # gat:g features
        g2 = g # h_func的shape： N x 768 (N:The sum of all nodes in the batch)
        h_func = g.ndata["_FUNC_EMB"] # shape N X 768 
        h = g.ndata["_UNIX_NODE_EMB"] # line_embedding: shape N X 768
        bboxes_feats = g.ndata['pos_emb']  # [N,4]

        h = self.gat(g, h) # h=node embedding vectors at current layer
        h = h.view(-1, h.size(1) * h.size(2)) 
        h = self.gat2(g, h) 
        h = h.view(-1, h.size(1) * h.size(2)) # [N,2048]
        h = self.mlpdropout(F.elu(self.fc(h))) # shape：N X  512
        h_func = self.mlpdropout(F.elu(self.fconly(h_func))) # shape：N X 512
        
        # Hidden layers
        for _, hlayer in enumerate(self.hidden):
            h = self.hdropout(F.elu(hlayer(h))) # elu(): SHAPE: N X  512
            h_func = self.hdropout(F.elu(hlayer(h_func))) # N X 512

        # unbatch_features
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
        GCN_img_emd = torch.mean(GCN_img_emd, dim=1) # 【bs，512】

        outputs = self.final_fc(GCN_img_emd)
        
        return outputs

# only-GCN
class Multi_DefectModel_Graph1(nn.Module):
    def __init__(self, config, pretrained=True, attention=True):
        super(Multi_DefectModel_Graph1, self).__init__()

        self.num_features = 1024 # self.swinV2.output_num()
        self.config = config
        self.num_classes = config.MODEL.NUM_CLASSES
        
        hfeat: int = 512 
        embfeat : int = 768 
        numheads = 4 # num_heads：Number of heads in Multi-Head Attention
        gatdrop: float = 0.2 # feat_drop：Dropout rate on feature
        mlpdropout: float = 0.2
        hdropout: float = 0.2
        ## node features mlp:
        self.fconly = th.nn.Linear(embfeat, hfeat)
        self.mlpdropout = th.nn.Dropout(0.2)
        ## Hidden Layers
        self.fch = []
        for _ in range(8):
            self.fch.append(th.nn.Linear(hfeat, hfeat))
        self.hidden = th.nn.ModuleList(self.fch)
        self.hdropout = th.nn.Dropout(hdropout)

        # GCN reasoning 
        self.max_node=100
        gcn_dim_size = 512 
        self.Rs_GCN_1 = Rs_GCN(in_channels=gcn_dim_size, inter_channels=gcn_dim_size)
        self.Rs_GCN_2 = Rs_GCN(in_channels=gcn_dim_size, inter_channels=gcn_dim_size)
        self.Rs_GCN_3 = Rs_GCN(in_channels=gcn_dim_size, inter_channels=gcn_dim_size)
        self.Rs_GCN_4 = Rs_GCN(in_channels=gcn_dim_size, inter_channels=gcn_dim_size)
        self.Rs_GCN_5 = Rs_GCN(in_channels=gcn_dim_size, inter_channels=gcn_dim_size)
        self.Rs_GCN_6 = Rs_GCN(in_channels=gcn_dim_size, inter_channels=gcn_dim_size)
        self.Rs_GCN_7 = Rs_GCN(in_channels=gcn_dim_size, inter_channels=gcn_dim_size)
        self.Rs_GCN_8 = Rs_GCN(in_channels=gcn_dim_size, inter_channels=gcn_dim_size)

        self.fc_gat = th.nn.Linear(hfeat, hfeat)
        self.bn_gat=nn.BatchNorm1d(self.max_node)

        self.bn_text = nn.BatchNorm1d(embfeat)
        self.ln_text = nn.LayerNorm(embfeat) 
        self.fc_text = th.nn.Linear(embfeat, hfeat)
       
        self.hbn = nn.BatchNorm1d(hfeat)
        self.hln = nn.LayerNorm(hfeat)
        self.hfc = th.nn.Linear(hfeat, hfeat)

        self.final_fc = th.nn.Linear(hfeat, self.num_classes) 
        self.final_fc_bn = nn.BatchNorm1d(hfeat*3) 

    def forward(self, g, img_embedding,func_text_embedding):
        
        h_func = g.ndata["_FUNC_EMB"] # shape N X 768 
        h = g.ndata["_UNIX_NODE_EMB"] # line_embedding: shape N X 768
        bboxes_feats = g.ndata['pos_emb']  # [N,4]
    
        h = self.mlpdropout(F.elu(self.fconly(h))) 
        h_func = self.mlpdropout(F.elu(self.fconly(h_func)))
        
        # Hidden layers
        for _, hlayer in enumerate(self.hidden):
            h = self.hdropout(F.elu(hlayer(h))) #  N X  512
            h_func = self.hdropout(F.elu(hlayer(h_func))) # N X 512

        g.ndata['HGATOUTPUT'] = h 
        g.ndata['HFGATOUTPUT'] = bboxes_feats 
        h_i,_,_=unbatch_features(g,self.max_node) # SHAPE: BS X MAX NODE X hfeat
        h_i = torch.stack(h_i)  # BS X MAX NODE X 512 (bsx100x512)

        h_i = F.elu(self.fc_gat(self.bn_gat(h_i))) # opt1 [bs,max node,512]
        # h_i = self.bn_gat(h_i) # opt2，
        GCN_img_emd = h_i # opt3
       
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
        
        outputs = self.final_fc(GCN_img_emd)

        return outputs

# only GAT
class Multi_DefectModel_Graph2(nn.Module):
    def __init__(self, config, pretrained=True, attention=True):
        super(Multi_DefectModel_Graph2, self).__init__()

        self.num_features = 1024 # self.swinV2.output_num()
        self.config = config
        self.num_classes = config.MODEL.NUM_CLASSES
        
        hfeat: int = 512 
        embfeat : int = 768 
        numheads = 4 
        gatdrop: float = 0.1 
        mlpdropout: float = 0.1
        hdropout: float = 0.1
        
        gnn_args = {"out_feats": hfeat} 
        
        gnn = GATConv
        gat_args = {"num_heads": numheads, "feat_drop": gatdrop} 
        gnn1_args = {**gnn_args, **gat_args, "in_feats": embfeat}  # embfeat：Input feature size
        gnn2_args = {**gnn_args, **gat_args, "in_feats": hfeat * numheads} #hfeat: int = 512,numheads=4    
        
        self.gat = gnn(**gnn1_args)
        self.gat2 = gnn(**gnn2_args)
        fcin = hfeat * numheads  # 4*512=2048
        self.fc = th.nn.Linear(fcin, hfeat) 
        self.fconly = th.nn.Linear(embfeat, hfeat)
        self.mlpdropout = th.nn.Dropout(mlpdropout)
        ## Hidden Layers
        self.fch = []
        for _ in range(8): 
            self.fch.append(th.nn.Linear(hfeat, hfeat))
        self.hidden = th.nn.ModuleList(self.fch)
        self.hdropout = th.nn.Dropout(hdropout)
        
        self.hbn = nn.BatchNorm1d(hfeat)
        self.hfc = th.nn.Linear(hfeat, hfeat)
        
        self.final_fc = th.nn.Linear(hfeat, self.num_classes) 
        
    def forward(self, g, img_embedding,func_text_embedding):
        
        g2 = g # N x 768
        h_func = g.ndata["_FUNC_EMB"] # shape N X 768 
        h = g.ndata["_UNIX_NODE_EMB"] # line_embedding: shape N X 768
        # h_all = g.ndata["_ALL_NODE_EMB"] # shape N X 800
        hdst=h

        gat = True # gat2layer
        if gat:
            h = self.gat(g, h) # h=node embedding vectors at current layer
            h = h.view(-1, h.size(1) * h.size(2)) 
            h = self.gat2(g2, h)
            h = h.view(-1, h.size(1) * h.size(2)) # 
            # shape: N X node X (hfeat * num_heads)
            h = self.mlpdropout(F.elu(self.fc(h))) # shape：N X  hfeat
            h_func = self.mlpdropout(F.elu(self.fconly(h_func))) # shape：N X hfeat
        
        else: # only-mlp
            h = self.mlpdropout(F.elu(self.fconly(hdst))) 
            h_func = self.mlpdropout(F.elu(self.fconly(h_func)))
            
        ## Hidden layers
        for _, hlayer in enumerate(self.hidden):
            h = self.hdropout(F.elu(hlayer(h))) 
            h_func = self.hdropout(F.elu(hlayer(h_func))) # N X hfeat

        with g.local_scope(): 
            g.ndata["h"] = h
            h_feature = dgl.mean_nodes(g,"h") # 【bs,512】
            h_feature = F.elu(self.hfc(self.hbn(h_feature))) 

        outputs = self.final_fc(h_feature)
        return outputs

