import time
import torch
import subprocess
import os
from model import EAST
from detect import detect_dataset,detect_dataset_map
import numpy as np
import shutil


def eval_model(model_name, test_img_path, submit_path, save_flag=True):
	if not os.path.exists(submit_path):
		# shutil.rmtree(submit_path) 
		os.makedirs(submit_path) # 

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = EAST(False).to(device)
	model.load_state_dict(torch.load(model_name))
	model.eval()
	
	start_time = time.time()
	# detect_dataset(model, device, test_img_path, submit_path)
	detect_dataset_map(model, device, test_img_path, submit_path)

	# os.chdir(submit_path)
	# res = subprocess.getoutput('zip -q submit.zip *.txt')
	# res = subprocess.getoutput('mv submit.zip ../')
	# os.chdir('../')

	# res = subprocess.getoutput('python3 ./evaluate/script.py –g=./evaluate/gt.zip –s=./submit.zip')
	# print(res)
	# os.remove('./submit.zip')
	print('eval time is {}'.format(time.time()-start_time))	

	# if not save_flag:
	# 	shutil.rmtree(submit_path)


if __name__ == '__main__': 
	model_name = './pths/model_epoch_190.pth'
	test_img_path = os.path.abspath('./imgs/pos_ast/test/buggy') # 
	test_img_path = os.path.abspath('./imgs/pos_ast/test/clean') # 
	test_img_path = os.path.abspath('./imgs/pos_ast/train/buggy') # 
	test_img_path = os.path.abspath('./imgs/pos_ast/train/clean') # 
	test_img_path = os.path.abspath('./imgs/pos_ast/valid/buggy') # 
	test_img_path = os.path.abspath('./imgs/pos_ast/valid/clean') # 
	
	submit_path = './norm_pos_dict' # Normalized coordinates
	eval_model(model_name, test_img_path, submit_path)
