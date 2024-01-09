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

class Multi_DefectModel_new_GCN(nn.Module):
    '''best modle'''
    def __init__(self, config, pretrained=True, attention=True):
        super(Multi_DefectModel_new_GCN, self).__init__()

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
        
        self.final_fc = th.nn.Linear(hfeat*3, self.num_classes) 
        self.final_fc_bn = nn.BatchNorm1d(hfeat*3) 

    def forward(self, g, img_embedding,func_text_embedding):
        
        ## 1.img feature shape: BS X 1024 ---> BS X 521
        x = self.swinfc(self.swinbn(img_embedding)) 
        x = F.elu(x) # nn.GELU x = F.gelu(x)
        # x =F.dropout(x, p=0.2, training=self.training)   

        ## 2.func_text_embedding
        func_text_embedding = self.bn_text(func_text_embedding)
        func_text_embedding = F.elu(self.fc_text(func_text_embedding))

        
        g2 = g 
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
            h = self.hdropout(F.elu(hlayer(h))) #  SHAPE: N X  512
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
        # print(f"GCN_img_emd.shape={GCN_img_emd.shape}")
        # GCN_img_emd = self.gcn_bn(pre_node_emb + GCN_img_emd) 
        GCN_img_emd = torch.mean(GCN_img_emd, dim=1) # 【bs，512】
        h_feature = GCN_img_emd

        all_feats = torch.cat((x, h_feature,func_text_embedding), dim=1) 
        # all_feats = torch.cat((x, h_feature,unix_feature), dim=1) 
        all_feats = self.final_fc(self.final_fc_bn(all_feats)) 
        # all_feats =F.dropout(self.final_fc(self.final_fc_bn(all_feats)), p=0.2, training=self.training) 
        return all_feats

# rq3: 010
class Multi_DefectModel(nn.Module):
    def __init__(self, config, pretrained=True, attention=True):
        super(Multi_DefectModel, self).__init__()

        self.num_features = 1024 # self.swinV2.output_num()
        self.config = config
        self.num_classes = config.MODEL.NUM_CLASSES
        hfeat: int = 512 
        embfeat : int = 768 
        numheads = 4 # num_heads：Number of heads in Multi-Head Attention

        gatdrop: float = 0.1 # feat_drop：Dropout rate on feature
        mlpdropout: float = 0.1
        hdropout: float = 0.1
        
        gnn_args = {"out_feats": hfeat} # Output feature size
        
        gnn = GATConv
        gat_args = {"num_heads": numheads, "feat_drop": gatdrop} 
        gnn1_args = {**gnn_args, **gat_args, "in_feats": embfeat}  # embfeat：Input feature size
        gnn2_args = {**gnn_args, **gat_args, "in_feats": hfeat * numheads} #hfeat: int = 512,numheads=4    
        ## model：gat2layer:
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
        ## text embedding
        self.bn_text = nn.BatchNorm1d(embfeat) 
        self.fc_text = th.nn.Linear(embfeat, hfeat)
        ## swin img embedding
        self.swinbn = nn.BatchNorm1d(self.num_features)
        self.swinfc = th.nn.Linear(self.num_features, hfeat)
        ## h_feature：
        self.hbn = nn.BatchNorm1d(hfeat)
        self.hfc = th.nn.Linear(hfeat, hfeat)
        
        self.final_fc = th.nn.Linear(hfeat*3, self.num_classes) 
        self.final_fc_bn = nn.BatchNorm1d(hfeat*3) 

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
        # h_all = g.ndata["_ALL_NODE_EMB"] # shape N X 800
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
        
        # Hidden layers
        for _, hlayer in enumerate(self.hidden):
            h = self.hdropout(F.elu(hlayer(h))) 
            h_func = self.hdropout(F.elu(hlayer(h_func))) # N X hfeat

        with g.local_scope(): 
            g.ndata["h"] = h
            h_feature = dgl.mean_nodes(g,"h") # 【bs,512】
            h_feature = F.elu(self.hfc(self.hbn(h_feature))) 

        all_feats = torch.cat((x, h_feature,func_text_embedding), dim=1) 
        all_feats = self.final_fc(self.final_fc_bn(all_feats)) 
        
        return all_feats 

class Multi_DefectModel_noGraph(nn.Module):
    def __init__(self, config, pretrained=True, attention=True):
        super(Multi_DefectModel_noGraph, self).__init__()

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
        ## swin img embedding
        self.swinbn = nn.BatchNorm1d(self.num_features)
        self.swinfc = th.nn.Linear(self.num_features, hfeat)
        ## h_feature：
        self.hbn = nn.BatchNorm1d(hfeat)
        self.hln = nn.LayerNorm(hfeat)
        self.hfc = th.nn.Linear(hfeat, hfeat)
        
        self.final_fc = th.nn.Linear(hfeat*2, self.num_classes) 
        self.final_fc_bn = nn.BatchNorm1d(hfeat*2) 

    def forward(self, g, img_embedding,func_text_embedding):
        
        ## 1.img feature shape: BS X 1024 ---> BS X 521
        x = self.swinfc(self.swinbn(img_embedding)) 
        x = F.elu(x) # nn.GELU x = F.gelu(x)
        # x =F.dropout(x, p=0.2, training=self.training)  

        ## 2.func_text_embedding
        func_text_embedding = self.bn_text(func_text_embedding)
        func_text_embedding = F.elu(self.fc_text(func_text_embedding))

        # all_feats = torch.cat((x, h_feature,func_text_embedding), dim=1) 
        all_feats = torch.cat((x,func_text_embedding), dim=1)
        outputs = self.final_fc(self.final_fc_bn(all_feats))  
        return outputs


class Multi_DefectModel_000(nn.Module):
    def __init__(self, config, pretrained=True, attention=True):
        super(Multi_DefectModel_000, self).__init__()

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

        
        self.bn_text = nn.BatchNorm1d(embfeat)
        self.ln_text = nn.LayerNorm(embfeat) 
        self.fc_text = th.nn.Linear(embfeat, hfeat)
        
        self.swinbn = nn.BatchNorm1d(self.num_features)
        self.swinfc = th.nn.Linear(self.num_features, hfeat)
       
        self.hbn = nn.BatchNorm1d(hfeat)
        self.hln = nn.LayerNorm(hfeat)
        self.hfc = th.nn.Linear(hfeat, hfeat)
        
        self.final_fc = th.nn.Linear(hfeat*3, self.num_classes) 
        self.final_fc_bn = nn.BatchNorm1d(hfeat*3) 

    def forward(self, g, img_embedding,func_text_embedding):
        
        ## 1.img feature shape: BS X 1024 ---> BS X 521
        x = self.swinfc(self.swinbn(img_embedding)) 
        x = F.elu(x) # nn.GELU x = F.gelu(x)
        # x =F.dropout(x, p=0.2, training=self.training)  

        ## 2.func_text_embedding
        func_text_embedding = self.bn_text(func_text_embedding)
        func_text_embedding = F.elu(self.fc_text(func_text_embedding))

        ## 3.node features
        h_func = g.ndata["_FUNC_EMB"] # shape N X 768 
        h = g.ndata["_UNIX_NODE_EMB"] # line_embedding: shape N X 768
        # mlp 
        h = self.mlpdropout(F.elu(self.fconly(h))) 
        h_func = self.mlpdropout(F.elu(self.fconly(h_func)))
        # Hidden layers
        # for _, hlayer in enumerate(self.hidden):
        #     h = self.hdropout(F.elu(hlayer(h))) 
        #     h_func = self.hdropout(F.elu(hlayer(h_func))) # N X 512

        with g.local_scope(): 
            g.ndata["h"] = h
            h_feature = dgl.mean_nodes(g,"h") # 【bs,512】
            h_feature = F.elu(self.hfc(self.hbn(h_feature))) 

        all_feats = torch.cat((x, h_feature,func_text_embedding), dim=1) 

        outputs = self.final_fc(self.final_fc_bn(all_feats))  
        return outputs

class Multi_DefectModel_001(nn.Module):
    def __init__(self, config, pretrained=True, attention=True):
        super(Multi_DefectModel_001, self).__init__()

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
        
        self.swinbn = nn.BatchNorm1d(self.num_features)
        self.swinfc = th.nn.Linear(self.num_features, hfeat)
        
        self.hbn = nn.BatchNorm1d(hfeat)
        self.hln = nn.LayerNorm(hfeat)
        self.hfc = th.nn.Linear(hfeat, hfeat)
        
        self.final_fc = th.nn.Linear(hfeat*3, self.num_classes) 
        self.final_fc_bn = nn.BatchNorm1d(hfeat*3) 

    def forward(self, g, img_embedding,func_text_embedding):
        
        x = self.swinfc(self.swinbn(img_embedding)) 
        x = F.elu(x) # nn.GELU x = F.gelu(x)
        
        func_text_embedding = self.bn_text(func_text_embedding)
        func_text_embedding = F.elu(self.fc_text(func_text_embedding))

        h_func = g.ndata["_FUNC_EMB"] # shape N X 768 
        h = g.ndata["_UNIX_NODE_EMB"] # line_embedding: shape N X 768
        bboxes_feats = g.ndata['pos_emb']  # [N,4]
        # mlp 
        h = self.mlpdropout(F.elu(self.fconly(h))) 
        h_func = self.mlpdropout(F.elu(self.fconly(h_func)))
        # Hidden layers
        # for _, hlayer in enumerate(self.hidden):
        #     h = self.hdropout(F.elu(hlayer(h))) 
        #     h_func = self.hdropout(F.elu(hlayer(h_func))) # N X 512
        
        g.ndata['HGATOUTPUT'] = h 
        g.ndata['HFGATOUTPUT'] = bboxes_feats 
        h_i,_,_=unbatch_features(g,self.max_node) 
        h_i = torch.stack(h_i)  # BS X MAX NODE X 512 (bsx100x512)

        h_i = F.elu(self.fc_gat(self.bn_gat(h_i))) # [bs,max node,512] 
        # h_i = self.bn_gat(h_i) 
        GCN_img_emd = h_i # 
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
    
        all_feats = torch.cat((x, h_feature,func_text_embedding), dim=1) 

        outputs = self.final_fc(self.final_fc_bn(all_feats))  
        return outputs

# rq3: 100：
class Multi_DefectModel_100(nn.Module):
    def __init__(self, config, pretrained=True, attention=True):
        super(Multi_DefectModel_100, self).__init__()

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
        self.max_node=100
        ## text embedding
        self.bn_text = nn.BatchNorm1d(embfeat)
        self.ln_text = nn.LayerNorm(embfeat) 
        self.fc_text = th.nn.Linear(embfeat, hfeat)
        ## swin img embedding
        self.swinbn = nn.BatchNorm1d(self.num_features)
        self.swinfc = th.nn.Linear(self.num_features, hfeat)
        ## h_feature：
        self.bn_gat = nn.BatchNorm1d(self.max_node) 
        self.fc_gat = th.nn.Linear(512, 480)
        self.bn_bbox = nn.BatchNorm1d(self.max_node) 
        self.fc_bbox = th.nn.Linear(4, 32)
        self.hbn = nn.BatchNorm1d(hfeat)
        self.hln = nn.LayerNorm(hfeat)
        self.hfc = th.nn.Linear(hfeat, hfeat)
        
        self.final_fc = th.nn.Linear(hfeat*3, self.num_classes) 
        self.final_fc_bn = nn.BatchNorm1d(hfeat*3) 

    def forward(self, g, img_embedding,func_text_embedding):
        
        ## 1.img feature shape: BS X 1024 ---> BS X 521
        x = self.swinfc(self.swinbn(img_embedding)) 
        x = F.elu(x) # nn.GELU x = F.gelu(x)
        
        ## 2.func_text_embedding
        func_text_embedding = self.bn_text(func_text_embedding)
        func_text_embedding = F.elu(self.fc_text(func_text_embedding))

        ## 3.node features
        h_func = g.ndata["_FUNC_EMB"] # shape N X 768 
        h = g.ndata["_UNIX_NODE_EMB"] # line_embedding: shape N X 768
        # mlp 
        h = self.mlpdropout(F.elu(self.fconly(h))) # N x 512
        h_func = self.mlpdropout(F.elu(self.fconly(h_func)))
        
        # Hidden layers
        # for _, hlayer in enumerate(self.hidden):
        #     h = self.hdropout(F.elu(hlayer(h))) # elu():激活函数 SHAPE: N X  512
        #     h_func = self.hdropout(F.elu(hlayer(h_func))) # N X 512
        
        bboxes_feats = g.ndata['pos_emb']  # [N,4]

        g.ndata['HGATOUTPUT'] = h 
        g.ndata['HFGATOUTPUT'] = bboxes_feats 
        h_i,hf_i,unix_i=unbatch_features(g,self.max_node) # SHAPE: BS X MAX NODE X hfeat
        h_i = torch.stack(h_i)  # BS X MAX NODE X 512 (bsx100x512)
        pos_i = torch.stack(hf_i) # BS X MAX NODE X 4
    
        h_i = F.elu(self.fc_gat(self.bn_gat(h_i))) # [bs,max node,480]
        pos_i = F.elu(self.fc_bbox(self.bn_bbox(pos_i))) # [bs,max node,32]
        GCN_img_emd = torch.cat((h_i, pos_i), dim=2)  # [bs,max node,512] 
        h_feature = torch.mean(GCN_img_emd, dim=1) # 【bs，512】

        all_feats = torch.cat((x, h_feature,func_text_embedding), dim=1) 

        outputs = self.final_fc(self.final_fc_bn(all_feats))  
        return outputs

# rq3: 110 gat+pos
class Multi_DefectModel_110(nn.Module):
    def __init__(self, config, pretrained=True, attention=True):
        super(Multi_DefectModel_110, self).__init__()

        self.num_features = 1024 
        self.config = config
        self.num_classes = config.MODEL.NUM_CLASSES
    
        hfeat: int = 512 
        embfeat : int = 768 
        numheads = 4 # num_heads：Number of heads in Multi-Head Attention

        gatdrop: float = 0.1 # feat_drop：Dropout rate on feature
        mlpdropout: float = 0.1
        hdropout: float = 0.1
        
        gnn_args = {"out_feats": hfeat} # Output feature size
        
        gnn = GATConv
        gat_args = {"num_heads": numheads, "feat_drop": gatdrop} 
        gnn1_args = {**gnn_args, **gat_args, "in_feats": embfeat}  # embfeat：Input feature size
        gnn2_args = {**gnn_args, **gat_args, "in_feats": hfeat * numheads} #hfeat: int = 512,numheads=4    
        ## model：gat2layer:
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
        
        self.bn_text = nn.BatchNorm1d(embfeat) 
        self.fc_text = th.nn.Linear(embfeat, hfeat)
        
        self.swinbn = nn.BatchNorm1d(self.num_features)
        self.swinfc = th.nn.Linear(self.num_features, hfeat)
        
        self.max_node=100
        self.bn_gat = nn.BatchNorm1d(self.max_node) 
        self.fc_gat = th.nn.Linear(512, 480)
        self.bn_bbox = nn.BatchNorm1d(self.max_node) 
        self.fc_bbox = th.nn.Linear(4, 32)
         
        self.final_fc = th.nn.Linear(hfeat*3, self.num_classes) 
        self.final_fc_bn = nn.BatchNorm1d(hfeat*3) 

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
        # h_all = g.ndata["_ALL_NODE_EMB"] # shape N X 800
        hdst=h

        gat = True 
        if gat:
            h = self.gat(g, h) # h=node embedding vectors at current layer
            h = h.view(-1, h.size(1) * h.size(2)) 
            h = self.gat2(g2, h)
            h = h.view(-1, h.size(1) * h.size(2))  
            h = self.mlpdropout(F.elu(self.fc(h))) # shape：N X  hfeat
            h_func = self.mlpdropout(F.elu(self.fconly(h_func))) # shape：N X hfeat
        else: # only-mlp
            h = self.mlpdropout(F.elu(self.fconly(hdst))) 
            h_func = self.mlpdropout(F.elu(self.fconly(h_func)))
        
        ## Hidden layers
        for _, hlayer in enumerate(self.hidden):
            h = self.hdropout(F.elu(hlayer(h))) 
            h_func = self.hdropout(F.elu(hlayer(h_func))) # N X hfeat
        ## pos
        bboxes_feats = g.ndata['pos_emb']  # [N,4]
        # unbatch_features
        g.ndata['HGATOUTPUT'] = h 
        g.ndata['HFGATOUTPUT'] = bboxes_feats 
        h_i,hf_i,unix_i=unbatch_features(g,self.max_node) # SHAPE: BS X MAX NODE X hfeat
        h_i = torch.stack(h_i)  # BS X MAX NODE X 512 (bsx100x512)
        pos_i = torch.stack(hf_i) # BS X MAX NODE X 4
    
        h_i = F.elu(self.fc_gat(self.bn_gat(h_i))) # [bs,max node,480]
        pos_i = F.elu(self.fc_bbox(self.bn_bbox(pos_i))) # [bs,max node,32]
        GCN_img_emd = torch.cat((h_i, pos_i), dim=2)  # [bs,max node,512] 
        h_feature = torch.mean(GCN_img_emd, dim=1) # 【bs，512】
    
        all_feats = torch.cat((x, h_feature,func_text_embedding), dim=1) 
        all_feats = self.final_fc(self.final_fc_bn(all_feats))  

        return all_feats 

# rq3: put pos into gat(110
class Multi_DefectModel_GATPOS(nn.Module):
    def __init__(self, config, pretrained=True, attention=True):
        super(Multi_DefectModel_GATPOS, self).__init__()

        self.num_features = 1024 # self.swinV2.output_num()
        self.config = config
        self.num_classes = config.MODEL.NUM_CLASSES
        
        hfeat: int = 512
        embfeat : int = 768 
        numheads = 4 # num_heads：Number of heads in Multi-Head Attention

        gatdrop: float = 0.1 # feat_drop：Dropout rate on feature
        mlpdropout: float = 0.1
        hdropout: float = 0.1
        
        gnn_args = {"out_feats": hfeat} # Output feature size
        
        gnn = GATConv
        gat_args = {"num_heads": numheads, "feat_drop": gatdrop} 
        gnn1_args = {**gnn_args, **gat_args, "in_feats": embfeat}  # embfeat：Input feature size
        gnn2_args = {**gnn_args, **gat_args, "in_feats": hfeat * numheads} #hfeat: int = 512,numheads=4    
        ## model：gat2layer:
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
        
        self.bn_text = nn.BatchNorm1d(embfeat) 
        self.fc_text = th.nn.Linear(embfeat, hfeat)
       
        self.swinbn = nn.BatchNorm1d(self.num_features)
        self.swinfc = th.nn.Linear(self.num_features, hfeat)
        
        self.hbn = nn.BatchNorm1d(hfeat)
        self.hfc = th.nn.Linear(hfeat, hfeat)
        self.max_node=100
        self.bn_gat = nn.BatchNorm1d(self.max_node) 
        self.fc_gat = th.nn.Linear(768, 720)
        self.bn_bbox = nn.BatchNorm1d(self.max_node) 
        self.fc_bbox = th.nn.Linear(4, 48)
        
        self.final_fc = th.nn.Linear(hfeat*3, self.num_classes) 
        self.final_fc_bn = nn.BatchNorm1d(hfeat*3) 

    def forward(self, g, img_embedding,func_text_embedding):
        
        ## 1.img feature shape: BS X 1024 ---> BS X 521
        x = self.swinfc(self.swinbn(img_embedding)) 
        x = F.elu(x)

        ## 2.func_text_embedding
        func_text_embedding = self.bn_text(func_text_embedding)
        func_text_embedding = F.elu(self.fc_text(func_text_embedding))

        g2 = g # h_func shape： N x 768 
        h_func = g.ndata["_FUNC_EMB"] # shape N X 768 
        h = g.ndata["_UNIX_NODE_EMB"] # line_embedding: shape N X 768
        bboxes_feats = g.ndata['pos_emb']  #[N,4]
        hdst=h

        h = F.elu(self.fc_gat(h)) # [N,720] 
        bboxes_feats = F.elu(self.fc_bbox(bboxes_feats)) # [N,48]
        h = torch.cat((h, bboxes_feats), dim=1) # [N,768]

        gat = True 
        if gat:
            h = self.gat(g, h) # h=node embedding vectors at current layer
            h = h.view(-1, h.size(1) * h.size(2)) 
            h = self.gat2(g2, h) 
            h = h.view(-1, h.size(1) * h.size(2)) 
            h = self.mlpdropout(F.elu(self.fc(h))) # shape：N X  hfeat
            h_func = self.mlpdropout(F.elu(self.fconly(h_func))) # shape：N X hfeat
        else: # only-mlp
            h = self.mlpdropout(F.elu(self.fconly(hdst))) 
            h_func = self.mlpdropout(F.elu(self.fconly(h_func)))
        
        ## Hidden layers
        for _, hlayer in enumerate(self.hidden):
            h = self.hdropout(F.elu(hlayer(h))) 
            h_func = self.hdropout(F.elu(hlayer(h_func))) # N X 512
        
        g.ndata['HGATOUTPUT'] = h 
        g.ndata['HFGATOUTPUT'] = bboxes_feats 
        h_i,_,_=unbatch_features(g,self.max_node) # SHAPE: BS X MAX NODE X hfeat
        h_i = torch.stack(h_i)  
        h_i = F.elu(self.hfc(self.bn_gat(h_i))) # [bs,max node,512]:
        h_feature = torch.mean(h_i, dim=1) # 【bs，512】
        
        # with g.local_scope(): 
        #     g.ndata["h"] = h
        #     h_feature = dgl.mean_nodes(g,"h") # 【bs,512】
        #     h_feature = F.elu(self.hfc(self.hbn(h_feature))) 

        all_feats = torch.cat((x, h_feature,func_text_embedding), dim=1) 
        all_feats = self.final_fc(self.final_fc_bn(all_feats))  

        return all_feats 


# rq3: 011 : gat+gcn
class Multi_DefectModel_011(nn.Module):
    def __init__(self, config, pretrained=True, attention=True):
        super(Multi_DefectModel_011, self).__init__()

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
        self.fc_gat = th.nn.Linear(hfeat, hfeat)
        self.bn_gat=nn.BatchNorm1d(self.max_node)
        
        self.swinbn = nn.BatchNorm1d(self.num_features)
        self.swinfc = th.nn.Linear(self.num_features, hfeat)
       
        self.hbn = nn.BatchNorm1d(hfeat)
        self.hln = nn.LayerNorm(hfeat)
        self.hfc = th.nn.Linear(hfeat, hfeat)
    
        self.final_fc = th.nn.Linear(hfeat*3, self.num_classes) 
        self.final_fc_bn = nn.BatchNorm1d(hfeat*3) 

    def forward(self, g, img_embedding,func_text_embedding):
        
        ## 1.img feature shape: BS X 1024 ---> BS X 521
        x = self.swinfc(self.swinbn(img_embedding)) 
        x = F.elu(x) # nn.GELU x = F.gelu(x)
       
        ## 2.func_text_embedding
        func_text_embedding = self.bn_text(func_text_embedding)
        func_text_embedding = F.elu(self.fc_text(func_text_embedding))

        g2 = g # h_func的shape： N x 768
        h_func = g.ndata["_FUNC_EMB"] # shape N X 768 
        h = g.ndata["_UNIX_NODE_EMB"] # line_embedding: shape N X 768
        bboxes_feats = g.ndata['pos_emb']  # [N,4]

        h = self.gat(g, h) # h=node embedding vectors at current layer
        h = h.view(-1, h.size(1) * h.size(2)) 
        h = self.gat2(g, h) 
        h = h.view(-1, h.size(1) * h.size(2)) # [N,2048]
        h = self.mlpdropout(F.elu(self.fc(h))) # shape：N X  512
        h_func = self.mlpdropout(F.elu(self.fconly(h_func))) # shape：N X 512
        
        ## Hidden layers
        for _, hlayer in enumerate(self.hidden):
            h = self.hdropout(F.elu(hlayer(h))) # elu()
            h_func = self.hdropout(F.elu(hlayer(h_func))) # N X 512

        g.ndata['HGATOUTPUT'] = h 
        g.ndata['HFGATOUTPUT'] = bboxes_feats 
        h_i,pos_i,_=unbatch_features(g,self.max_node) # SHAPE: BS X MAX NODE X hfeat
        h_i = torch.stack(h_i)  # BS X MAX NODE X 512 (bsx100x512)
        # pos_i = torch.stack(pos_i) # BS X MAX NODE X 4

        # h_i = F.elu(self.fc_gat(self.bn_gat(h_i))) 
        h_i = F.elu(self.bn_gat(h_i)) 
        GCN_img_emd = h_i
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

        all_feats = torch.cat((x, h_feature,func_text_embedding), dim=1) 
        all_feats = self.final_fc(self.final_fc_bn(all_feats))
        
        return all_feats

class Multi_DefectModel_NOGAT(nn.Module):
    '''text embedding不经过gat直接进入gcn看看效果'''
    def __init__(self, config, pretrained=True, attention=True):
        super(Multi_DefectModel_NOGAT, self).__init__()

        self.num_features = 1024 # self.swinV2.output_num()
        self.config = config
        self.num_classes = config.MODEL.NUM_CLASSES
        
        hfeat: int = 512 
        embfeat : int = 768 
        numheads = 4 # num_heads：Number of heads in Multi-Head Attention
        gatdrop: float = 0.2 # feat_drop：Dropout rate on feature
        mlpdropout: float = 0.2
        hdropout: float = 0.2
        self.mlpdropout = th.nn.Dropout(mlpdropout)

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
        self.fc_gat = th.nn.Linear(768, 480)
        # self.bn_gat1 = nn.BatchNorm1d(self.max_node)
        # self.fc_gat1 = th.nn.Linear(768, 521)
        # self.bn_gat2 = nn.BatchNorm1d(self.max_node) 
        # self.fc_gat2 = th.nn.Linear(521, 480)

        self.bn_bbox = nn.BatchNorm1d(self.max_node) 
        self.fc_bbox = th.nn.Linear(4, 32)
       
        self.swinbn = nn.BatchNorm1d(self.num_features)
        self.swinfc = th.nn.Linear(self.num_features, hfeat)
        
        self.hbn = nn.BatchNorm1d(hfeat)
        self.hln = nn.LayerNorm(hfeat)
        self.hfc = th.nn.Linear(hfeat, hfeat)
        
        self.final_fc = th.nn.Linear(hfeat*3, self.num_classes) 
        self.final_fc_bn = nn.BatchNorm1d(hfeat*3) 

    def forward(self, g, img_embedding,func_text_embedding):
        
        ## 1.img feature shape: BS X 1024 ---> BS X 521
        x = self.swinfc(self.swinbn(img_embedding)) 
        x = F.elu(x)
        # x =F.dropout(x, p=0.2, training=self.training)  

        ## 2.func_text_embedding
        func_text_embedding = self.bn_text(func_text_embedding)
        func_text_embedding = F.elu(self.fc_text(func_text_embedding))

        h = g.ndata["_UNIX_NODE_EMB"] # 【N ，768】
        bboxes_feats = g.ndata['pos_emb']  # [N,4]

        g.ndata['HGATOUTPUT'] = h 
        g.ndata['HFGATOUTPUT'] = bboxes_feats 
        h_i,pos_i,_=unbatch_features(g,self.max_node) 
        h_i = torch.stack(h_i)  # BS X MAX NODE X 768
        pos_i = torch.stack(pos_i) # BS X MAX NODE X 4
       
        h_i = F.elu(self.fc_gat(self.bn_gat(h_i))) # [bs,max node,480]
        
        # h_i = F.elu(self.fc_gat1(self.bn_gat1(h_i))) # [bs,max node,512]
        # h_i = F.elu(self.fc_gat2(self.bn_gat2(h_i))) # [bs,max node,480]

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

        all_feats = torch.cat((x, h_feature,func_text_embedding), dim=1) 
        all_feats = self.final_fc(self.final_fc_bn(all_feats)) 
        
        return all_feats

# rq3: 101:
class Multi_DefectModel_NOGAT3(nn.Module):
    def __init__(self, config, pretrained=True, attention=True):
        super(Multi_DefectModel_NOGAT3, self).__init__()

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

        self.pos_fch = []
        for _ in range(8): 
            self.pos_fch.append(th.nn.Linear(128, 128))
        self.pos_hidden = th.nn.ModuleList(self.pos_fch)

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
        self.fc_bbox = th.nn.Linear(4, 128)
        self.fc_bbox2 = th.nn.Linear(128, 32)
       
        self.swinbn = nn.BatchNorm1d(self.num_features)
        self.swinfc = th.nn.Linear(self.num_features, hfeat)
        
        self.hbn = nn.BatchNorm1d(hfeat)
        self.hln = nn.LayerNorm(hfeat)
        self.hfc = th.nn.Linear(hfeat, hfeat)
        
        self.final_fc = th.nn.Linear(hfeat*3, self.num_classes) 
        self.final_fc_bn = nn.BatchNorm1d(hfeat*3) 

    def forward(self, g, img_embedding,func_text_embedding):
        
        ## 1.img feature shape: BS X 1024 ---> BS X 521
        x = self.swinfc(self.swinbn(img_embedding)) 
        x = F.elu(x) # nn.GELU x = F.gelu(x)
        # x =F.dropout(x, p=0.2, training=self.training)  

        ## 2.func_text_embedding
        func_text_embedding = self.bn_text(func_text_embedding)
        func_text_embedding = F.elu(self.fc_text(func_text_embedding))

        ## 3.node features
        h_func = g.ndata["_FUNC_EMB"] # shape N X 768 
        h = g.ndata["_UNIX_NODE_EMB"] # line_embedding: shape N X 768
        bboxes_feats = g.ndata['pos_emb']  # [N,4]
        # mlp 
        h = self.mlpdropout(F.elu(self.fconly(h))) 
        h_func = self.mlpdropout(F.elu(self.fconly(h_func)))
        bboxes_feats= F.elu(self.fc_bbox(bboxes_feats)) # [bs,max node,128]
        # Hidden layers
        for _, hlayer in enumerate(self.hidden):
            h = self.hdropout(F.elu(hlayer(h))) 
            h_func = self.hdropout(F.elu(hlayer(h_func))) # N X 512
        # bbox hidden layers
        for _, pos_layer in enumerate(self.pos_hidden):
            bboxes_feats = self.hdropout(F.elu(pos_layer(bboxes_feats))) # SHAPE: N X 128

        g.ndata['HGATOUTPUT'] = h 
        g.ndata['HFGATOUTPUT'] = bboxes_feats 
        h_i,pos_i,_=unbatch_features(g,self.max_node) 
        h_i = torch.stack(h_i)  # BS X MAX NODE X 512 (bsx100x512)
        pos_i = torch.stack(pos_i) # BS X MAX NODE X 128
    
        h_i = F.elu(self.fc_gat(self.bn_gat(h_i))) # [bs,max node,480]
        pos_i = F.elu(self.fc_bbox2(self.bn_bbox(pos_i))) # [bs,max node,32]

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

        all_feats = torch.cat((x, h_feature,func_text_embedding), dim=1) 
        outputs = self.final_fc(self.final_fc_bn(all_feats)) 
        
        return outputs

# rq3: 101: 
class Multi_DefectModel_NOGAT4(nn.Module):
    def __init__(self, config, pretrained=True, attention=True):
        super(Multi_DefectModel_NOGAT4, self).__init__()

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
        self.fconly = th.nn.Linear(embfeat, 480)
        self.mlpdropout = th.nn.Dropout(0.2)
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
        self.fc_gat = th.nn.Linear(512, 512)
        self.fc_bbox = th.nn.Linear(4, 32)
        
        self.swinbn = nn.BatchNorm1d(self.num_features)
        self.swinfc = th.nn.Linear(self.num_features, hfeat)
        
        self.hbn = nn.BatchNorm1d(hfeat)
        self.hln = nn.LayerNorm(hfeat)
        self.hfc = th.nn.Linear(hfeat, hfeat)
        
        self.final_fc = th.nn.Linear(hfeat*3, self.num_classes) 
        self.final_fc_bn = nn.BatchNorm1d(hfeat*3) 

    def forward(self, g, img_embedding,func_text_embedding):
        
        ## 1.img feature shape: BS X 1024 ---> BS X 521
        x = self.swinfc(self.swinbn(img_embedding))
        x = F.elu(x) # nn.GELU x = F.gelu(x)

        ## 2.func_text_embedding
        func_text_embedding = self.bn_text(func_text_embedding)
        func_text_embedding = F.elu(self.fc_text(func_text_embedding))

        ## 3.node features
        h_func = g.ndata["_FUNC_EMB"] # shape N X 768 
        h = g.ndata["_UNIX_NODE_EMB"] # line_embedding: shape N X 768
        bboxes_feats = g.ndata['pos_emb']  # [N,4]
        # mlp 
        h = self.mlpdropout(F.elu(self.fconly(h))) # shape：N X 480 
        bboxes_feats= F.elu(self.fc_bbox(bboxes_feats)) # N x 32
        h = torch.cat((h, bboxes_feats), dim=1) # N x 512 
        # Hidden layers
        for _, hlayer in enumerate(self.hidden):
            h = self.hdropout(F.elu(hlayer(h))) # N X  512
        
        g.ndata['HGATOUTPUT'] = h 
        g.ndata['HFGATOUTPUT'] = g.ndata['pos_emb'] 
        h_i,_,_=unbatch_features(g,self.max_node) # SHAPE: BS X MAX NODE X hfeat
        h_i = torch.stack(h_i)  # BS X MAX NODE X 512 (bsx100x512)
    
        h_i = F.elu(self.fc_gat(self.bn_gat(h_i))) # [bs,max node,512]
        GCN_img_emd = h_i  # [bs,max node,512]
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

        all_feats = torch.cat((x, h_feature,func_text_embedding), dim=1) 
        outputs = self.final_fc(self.final_fc_bn(all_feats)) 
        
        return outputs


# rq3: 101
class Multi_DefectModel_NOGAT2(nn.Module):
    def __init__(self, config, pretrained=True, attention=True):
        super(Multi_DefectModel_NOGAT2, self).__init__()

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
        
        self.final_fc = th.nn.Linear(hfeat*3, self.num_classes) 
        self.final_fc_bn = nn.BatchNorm1d(hfeat*3) 

    def forward(self, g, img_embedding,func_text_embedding):
        
        x = self.swinfc(self.swinbn(img_embedding)) 
        x = F.elu(x) 
        # x =F.dropout(x, p=0.2, training=self.training) 

        func_text_embedding = self.bn_text(func_text_embedding)
        func_text_embedding = F.elu(self.fc_text(func_text_embedding))

        ## 3.node features
        h_func = g.ndata["_FUNC_EMB"] # shape N X 768 
        h = g.ndata["_UNIX_NODE_EMB"] # line_embedding: shape N X 768
        # mlp 
        h = self.mlpdropout(F.elu(self.fconly(h))) 
        h_func = self.mlpdropout(F.elu(self.fconly(h_func)))
        # Hidden layers
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
        GCN_img_emd = torch.mean(GCN_img_emd, dim=1) # 【bs，512】
        h_feature = GCN_img_emd

        all_feats = torch.cat((x, h_feature,func_text_embedding), dim=1) 
        # all_feats = torch.cat((x, h_feature,unix_feature), dim=1) 
        outputs = self.final_fc(self.final_fc_bn(all_feats)) 
        # all_feats =F.dropout(self.final_fc(self.final_fc_bn(all_feats)), p=0.2, training=self.training)
        return outputs