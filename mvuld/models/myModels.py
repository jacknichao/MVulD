# -*- coding: utf-8 -*-
from __future__ import print_function, division
import sys
sys.path.insert(0,'.')

import dgl
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import pdb
import torch as th
import os

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv, GraphConv

# Custom fusion modules
from .swin_transformer_v2 import *
from .build import *

from utils import load_checkpoint,auto_resume_helper, \
    reduce_tensor,resume_bestf1_helper,save_bestf1_checkpoint

def normalize(x):
    return x / x.norm(dim=1, keepdim=True)

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X

class Multi_DefectModel_allnode(nn.Module):
    def __init__(self, config, pretrained=True, attention=True):
        super(Multi_DefectModel_allnode, self).__init__()

        self.num_features = 1024 # self.swinV2.output_num()
        self.config = config
        self.num_classes = config.MODEL.NUM_CLASSES
        
        hfeat: int = 512 
        func_embfeat : int = 768 
        embfeat : int = 800 
        gatdrop: float = 0.2 
        numheads = 4 
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
        self.fconly = th.nn.Linear(func_embfeat, hfeat) # 用于g里的func_emb的
        self.mlpdropout = th.nn.Dropout(mlpdropout)
        ## Hidden Layers
        self.fch = []
        for _ in range(8): 
            self.fch.append(th.nn.Linear(hfeat, hfeat))
        self.hidden = th.nn.ModuleList(self.fch)
        self.hdropout = th.nn.Dropout(hdropout)
        ## text embedding
        self.bn_text = nn.BatchNorm1d(func_embfeat)
        self.fc_text = th.nn.Linear(func_embfeat, hfeat)
        ## swin的img embedding
        self.swinbn = nn.BatchNorm1d(self.num_features)
        self.swinfc = th.nn.Linear(self.num_features, hfeat)
        # self.swinfc = th.nn.Linear(self.num_features, 1024)

        self.hbn = nn.BatchNorm1d(hfeat)
        self.hfc = th.nn.Linear(hfeat, hfeat)
    
        self.final_fc = th.nn.Linear(hfeat*3, self.num_classes) 
        self.final_fc_bn = nn.BatchNorm1d(hfeat*3) 

    def forward(self, g, img_embedding,func_text_embedding):
        
        ## 1.img feature shape: BS X 1024 ---> BS X 521
        x = self.swinfc(self.swinbn(img_embedding)) # 
        x = F.elu(x)
        # x =F.dropout(x, p=0.2, training=self.training)  

        ## 2.func_text_embedding
        func_text_embedding = self.bn_text(func_text_embedding)
        func_text_embedding = F.elu(self.fc_text(func_text_embedding))

        
        g2 = g # h_func的shape： N x 768 
        h_func = g.ndata["_FUNC_EMB"] # shape N X 768 
        # h = g.ndata["_UNIX_NODE_EMB"] # line_embedding: shape N X 768
        h = g.ndata["_ALL_NODE_EMB"] # shape N X 800
        hdst=h

        gat = True 
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

        g.ndata['HGATOUTPUT'] = h 
        g.ndata['HFGATOUTPUT'] = h_func 
        h_i,hf_i=unbatch_features(g) 
        # SHAPE: BS X MAX NODE X hfeat
        
        h_i = torch.stack(h_i)
        hf_i = torch.stack(hf_i) 
        
        h_feature = torch.mean(h_i, dim=1) 
        h_feature=F.elu(self.hfc(self.hbn(h_feature)))
        hf_feature = torch.mean(hf_i, dim=1) #

        all_feats = torch.cat((x, h_feature,func_text_embedding), dim=1) 
        all_feats = self.final_fc(self.final_fc_bn(all_feats)) 
        # all_feats =F.dropout(self.final_fc(self.final_fc_bn(all_feats)), p=0.3, training=self.training) 
        
        return all_feats 
    
def unbatch_features(g):
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

class Multi_DefectModel_grudot(nn.Module):
    def __init__(self, config, pretrained=True, attention=True):
        super(Multi_DefectModel_grudot, self).__init__()

        self.num_features = 1024 
        self.config = config
        self.num_classes = config.MODEL.NUM_CLASSES
        hfeat: int = 512 
        embfeat : int = 768 
        gatdrop: float = 0.2 # feat_drop：Dropout rate on feature
        numheads = 4 # num_heads：Number of heads in Multi-Head Attention
        mlpdropout: float = 0.2
        hdropout: float = 0.2
        gnn_args = {"out_feats": hfeat} # Output feature size
        
        gnn = GATConv
        gat_args = {"num_heads": numheads, "feat_drop": gatdrop} 
        gnn1_args = {**gnn_args, **gat_args, "in_feats": embfeat}  
        gnn2_args = {**gnn_args, **gat_args, "in_feats": hfeat * numheads} 
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
        
        self.bn_text = nn.BatchNorm1d(embfeat)
        self.fc_text = th.nn.Linear(embfeat, hfeat)
        
        self.swinbn = nn.BatchNorm1d(self.num_features)
        self.swinfc = th.nn.Linear(self.num_features, hfeat)
        
        ## h_feature：
        self.hbn = nn.BatchNorm1d(hfeat)
        self.hfc = th.nn.Linear(hfeat, hfeat)
        ##project 
        self.gru_local = nn.GRU(hfeat, hfeat, 1, batch_first=True)
        ## fusion
        self.final_bn = nn.BatchNorm1d(hfeat*2) 
        self.final_fc = nn.Linear(hfeat*2, self.num_classes)

    def forward(self, g, img_embedding,func_text_embedding):
        
        ## 1.img feature shape: BS X 1024 ---> BS X 521
        x = self.swinfc(self.swinbn(img_embedding)) 
        x = F.elu(x)

        ## 2.func_text_embedding
        func_text_embedding = self.bn_text(func_text_embedding)
        func_text_embedding = F.elu(self.fc_text(func_text_embedding))

        g2 = g # h_func的shape： N x 768 
        h_func = g.ndata["_FUNC_EMB"] # shape N X 768 
        h = g.ndata["_UNIX_NODE_EMB"] # line_embedding: shape N X 768
        h_all = g.ndata["_ALL_NODE_EMB"] # shape N X 800
        hdst=h

        gat = True 
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

        g.ndata['HGATOUTPUT'] = h 
        g.ndata['HFGATOUTPUT'] = h_func 
        h_i,hf_i=self.unbatch_features(g) 
        # SHAPE: BS X MAX NODE X hfeat
        
        h_i = torch.stack(h_i)
        hf_i = torch.stack(hf_i) 
        
        ## gru project:
        _, hidden_state = self.gru_local(h_i)
        h_i = hidden_state[0] 
        h_i = F.elu(self.hfc(self.hbn(h_i)))
        ## dot fusion 
        x = x * h_i  # Shape: bs x hfeat
        all_feats = torch.cat((x, func_text_embedding), dim=1) 
        # all_feats = x * func_text_embedding # 
        # all_feats = self.final_fc(self.final_bn(all_feats)) # 
        all_feats = F.dropout(self.final_fc(self.final_bn(all_feats)), p=0.3, training=self.training) 
        
        return all_feats 
    
    def unbatch_features(self, g):
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


class Multi_DefectModel(nn.Module):
    def __init__(self, config, pretrained=True, attention=True):
        super(Multi_DefectModel, self).__init__()
        self.num_features = 1024 # self.swinV2.output_num()
        self.config = config
        self.num_classes = config.MODEL.NUM_CLASSES
        
        hfeat: int = 512 
        embfeat : int = 768 
        gatdrop: float = 0.2 
        numheads = 4 
        mlpdropout: float = 0.2
        hdropout: float = 0.2
        gnn_args = {"out_feats": hfeat} 
        
        gnn = GATConv
        gat_args = {"num_heads": numheads, "feat_drop": gatdrop} 
        gnn1_args = {**gnn_args, **gat_args, "in_feats": embfeat}  
        gnn2_args = {**gnn_args, **gat_args, "in_feats": hfeat * numheads} 
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
        
        self.bn_text = nn.BatchNorm1d(embfeat)
        self.fc_text = th.nn.Linear(embfeat, hfeat)
        
        self.swinbn = nn.BatchNorm1d(self.num_features)
        self.swinfc = th.nn.Linear(self.num_features, hfeat)
        
        self.hbn = nn.BatchNorm1d(hfeat)
        self.hfc = th.nn.Linear(hfeat, hfeat)

        self.projection_layer ='gru'  # 'gru'  'attention' 'mean'
        self.fusion = 'attention' # 'attention' 'dot' 'concat'
        
        # PROJECTION LAYER
        self.gru_local = nn.GRU(hfeat, hfeat, 1, batch_first=True)
        
        # FINAL FUSION BEFORE CLASSIFICATION :
        # mmdim: MULTIMODAL DIM:Size of the Inner Multimodal Embedding')
        if self.fusion == 'attention' or self.fusion == 'dot':
            # ATTENTION or DOT PRODUCT AS FUSION
            self.final_bn = nn.BatchNorm1d(hfeat*2) 
            self.final_fc = nn.Linear(hfeat*2, self.num_classes)

        elif self.fusion == 'concat':
            # CONCATENATION AS FUSION
            self.final_bn = nn.BatchNorm1d(hfeat * 3)
            self.final_fc = nn.Linear(hfeat * 3, self.num_classes)
        else:
            print("Error: Last Layer Fusion selected not implemented")
        

    def forward(self, g, img_embedding,func_text_embedding):
        
        ## 1.img feature shape: BS X 1024 ---> BS X 521
        x = self.swinfc(self.swinbn(img_embedding)) 
        x = F.elu(x)

        ## 2.func_text_embedding
        func_text_embedding = self.bn_text(func_text_embedding)
        func_text_embedding = F.elu(self.fc_text(func_text_embedding))

        
        g2 = g # h_func的shape： N x 768 
        h_func = g.ndata["_FUNC_EMB"] # shape N X 768 
        h = g.ndata["_UNIX_NODE_EMB"] # line_embedding: shape N X 768
        h_all = g.ndata["_ALL_NODE_EMB"] # shape N X 800
        hdst=h

        gat = True 
        if gat:
            h = self.gat(g, h) # h=node embedding vectors at current layer
            h = h.view(-1, h.size(1) * h.size(2)) 
            h = self.gat2(g2, h) 
            h = h.view(-1, h.size(1) * h.size(2))  
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

        g.ndata['HGATOUTPUT'] = h 
        g.ndata['HFGATOUTPUT'] = h_func 
        h_i,hf_i=self.unbatch_features(g) 
        # SHAPE: BS X MAX NODE X hfeat
        
        h_i = torch.stack(h_i) # gatoutput-node embedding
        hf_i = torch.stack(hf_i)
        
        sample_size = x.size(0) 
        
        if self.projection_layer == 'gru':
            # GRU VISUAL+TEXTUAL UNDERSTANDING
            rnn_img, hidden_state = self.gru_local(h_i)
            h_i = hidden_state[0] # Hidden state of last time step of i layer (in this case only one layer)
        elif self.projection_layer == 'attention':
            # ATTENTION 
            visual_atnn = torch.bmm(x.reshape(sample_size,1,512), h_i.permute(0,2,1))
            visual_atnn = F.leaky_relu(visual_atnn)
            visual_atnn = F.softmax(visual_atnn, dim=2)
            # Attention over Global Visual Features
            h_i = torch.bmm(visual_atnn, h_i).reshape(sample_size, -1)
        elif self.projection_layer == 'mean': 
            h_i = torch.mean(h_i, dim=1)
        h_i = F.elu(self.hfc(self.hbn(h_i)))

        # print("###############################")
        # print(h_i.shape)  # shape：bs x hfeat

        if self.fusion == 'attention':
            # ATTENTION AS FUSION:
            visual_atnn = x * h_i  # Elem-wise mult - Shape: bs x feat
            visual_atnn = torch.tanh(visual_atnn)
            visual_atnn = F.softmax(visual_atnn, dim=1)
            # Attention over Global Visual Features
            x = visual_atnn * h_i  
            all_feats = torch.cat((x, func_text_embedding), dim=1) # shape: bs x (hfeat*2)
            all_feats = self.final_fc(self.final_bn(all_feats)) 
            # all_feats = F.dropout(self.final_fc(self.final_bn(all_feats)), p=0.3, training=self.training)

        elif self.fusion == 'dot':
            # DOT PRODUCT AS FUSION：Elem-wise mult
            x = x * h_i  # Shape: bs x hfeat
            all_feats = torch.cat((x, func_text_embedding), dim=1) # shape: bs x (hfeat*2)
            all_feats = F.dropout(self.final_fc(self.final_bn(all_feats)), p=0.3, training=self.training) 

        elif self.fusion == 'concat':
            x = torch.cat((x, h_i,func_text_embedding), dim=1)  # Shape: bs x （hfeat*3）
            all_feats =F.dropout(self.final_fc(self.final_bn(x)), p=0.3, training=self.training)  

        return all_feats 
    
    def unbatch_features(self, g):
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

