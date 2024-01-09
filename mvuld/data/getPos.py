import os
import pickle
import torch
import numpy as np
'''use _id to obtain ocr获取的位置信息(startX,endX,endX,endY)'''

# _id = 174388
# pos_dict_path = os.path.abspath('../datasets/norm_pos_dict')
# # svd.cache_dir() / f"swinv2_method_level_try5/{_id}.pt"
# pos_dict_file = os.path.join(pos_dict_path, f"{_id}.pkl")
# print(pos_dict_file)
# with open(pos_dict_file, "rb") as tf:
#     pd = pickle.load(tf)
# print(pd)
# # print(type(pd))
# # print(torch.tensor(pd[1], dtype=torch.long))
# ## python getPos.py

# norm_pos_dict = pd
# node_num =10
# node_bboxes = np.zeros((node_num, 4)) 
# for i in range(1,node_num+1):
#     if norm_pos_dict.has_key(i): 
#         node_bboxes[i-1] = norm_pos_dict[i] 
