import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from model import EAST
import os
from dataset import get_rotate_mat
import numpy as np
import lanms
import evaluate.test_lnms as lnms
from imutils.object_detection import non_max_suppression
import pytesseract
import cv2
import re
import pickle

def resize_img(img):
	'''resize image to be divisible by 32
	'''
	w, h = img.size
	resize_w = w
	resize_h = h

	resize_h = resize_h if resize_h % 32 == 0 else int(resize_h / 32) * 32
	resize_w = resize_w if resize_w % 32 == 0 else int(resize_w / 32) * 32
	img = img.resize((resize_w, resize_h), Image.BILINEAR)
	ratio_h = resize_h / h
	ratio_w = resize_w / w

	return img, ratio_h, ratio_w


def load_pil(img):
	'''convert PIL Image to torch.Tensor
	'''
	t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])
	return t(img).unsqueeze(0) 


def is_valid_poly(res, score_shape, scale):
	'''check if the poly in image scope
	Input:
		res        : restored poly in original image
		score_shape: score map shape
		scale      : feature map -> image
	Output:
		True if valid
	'''
	cnt = 0
	for i in range(res.shape[1]):
		if res[0,i] < 0 or res[0,i] >= score_shape[1] * scale or \
           res[1,i] < 0 or res[1,i] >= score_shape[0] * scale:
			cnt += 1
	return True if cnt <= 1 else False


def restore_polys(valid_pos, valid_geo, score_shape, scale=4):
	'''restore polys from feature maps in given positions
	Input:
		valid_pos  : potential text positions <numpy.ndarray, (n,2)>
		valid_geo  : geometry in valid_pos <numpy.ndarray, (5,n)>
		score_shape: shape of score map
		scale      : image / feature map
	Output:
		restored polys <numpy.ndarray, (n,8)>, index
	'''
	polys = []
	index = []
	valid_pos *= scale
	d = valid_geo[:4, :] # 4 x N
	angle = valid_geo[4, :] # N,

	for i in range(valid_pos.shape[0]):
		x = valid_pos[i, 0]
		y = valid_pos[i, 1]
		y_min = y - d[0, i]
		y_max = y + d[1, i]
		x_min = x - d[2, i]
		x_max = x + d[3, i]
		rotate_mat = get_rotate_mat(-angle[i])
		
		temp_x = np.array([[x_min, x_max, x_max, x_min]]) - x
		temp_y = np.array([[y_min, y_min, y_max, y_max]]) - y
		coordidates = np.concatenate((temp_x, temp_y), axis=0)
		res = np.dot(rotate_mat, coordidates)
		res[0,:] += x
		res[1,:] += y
		
		if is_valid_poly(res, score_shape, scale):
			index.append(i)
			polys.append([res[0,0], res[1,0], res[0,1], res[1,1], res[0,2], res[1,2],res[0,3], res[1,3]])
	return np.array(polys), index


def get_boxes(score, geo, score_thresh=0.9, nms_thresh=0.01):
	'''get boxes from feature map
	Input:
		score       : score map from model <numpy.ndarray, (1,row,col)>
		geo         : geo map from model <numpy.ndarray, (5,row,col)>
		score_thresh: threshold to segment score map
		nms_thresh  : threshold in nms: 
	Output:
		boxes       : final polys <numpy.ndarray, (n,9)>
	'''
	score = score[0,:,:]
	xy_text = np.argwhere(score > score_thresh) # n x 2, format is [r, c]
	if xy_text.size == 0:
		return None

	xy_text = xy_text[np.argsort(xy_text[:, 0])]
	valid_pos = xy_text[:, ::-1].copy() # n x 2, [x, y]
	valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]] # 5 x n

	polys_restored, index = restore_polys(valid_pos, valid_geo, score.shape) 
	if polys_restored.size == 0:
		return None

	boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
	boxes[:, :8] = polys_restored
	boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]] 
	# print(len(boxes))
	# boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh) 
	boxes = lnms.nms_locality(boxes.astype('float32'), nms_thresh) 
	# print(len(boxes))
	return boxes

def adjust_ratio(boxes, ratio_w, ratio_h):
	'''refine boxes
	Input:
		boxes  : detected polys <numpy.ndarray, (n,9)>
		ratio_w: ratio of width
		ratio_h: ratio of height
	Output:
		refined boxes
	'''
	if boxes is None or boxes.size == 0:
		return None
	boxes[:,[0,2,4,6]] /= ratio_w
	boxes[:,[1,3,5,7]] /= ratio_h
	return np.around(boxes) 

def adjust_ratio_and_normalize(boxes, ratio_w, ratio_h,w,h):
	'''refine boxes and normalize by width and height of image
	refined and normalized boxes
	'''
	if boxes is None or boxes.size == 0:
		return None
	boxes[:,[0,2,4,6]] /= ratio_w
	boxes[:,[1,3,5,7]] /= ratio_h
	# print(boxes)
	boxes[:,[0,2,4,6]] /= w*1.0
	boxes[:,[1,3,5,7]] /= h*1.0
	# print(boxes)
	# return np.around(boxes)
	return boxes

def detect(img, model, device):
	'''detect text regions of img using model
	Input:
		img   : PIL Image
		model : detection model
		device: gpu if gpu is available
	Output:
		detected polys
	'''
	# img = img.convert("RGB") 
	w, h = img.size
	img, ratio_h, ratio_w = resize_img(img)
	with torch.no_grad():
		score, geo = model(load_pil(img).to(device))
	boxes = get_boxes(score.squeeze(0).cpu().numpy(), geo.squeeze(0).cpu().numpy())
	# print(boxes)
	return adjust_ratio(boxes, ratio_w, ratio_h)
	# return adjust_ratio_and_normalize(boxes, ratio_w, ratio_h,w,h) # 

def plot_boxes(img, boxes):
	'''plot boxes on image
	'''
	if boxes is None:
		print("boxes is none") # 
		return img
	
	draw = ImageDraw.Draw(img)
	for box in boxes: 
		draw.polygon([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]], outline=(0,255,0))
	return img

def plot_boxes_text(img_path, boxes,res_img):
	'''plot boxes and text on image
	'''
	image = cv2.imread(img_path)
	orig = image.copy()
	origH, origW = image.shape[:2]
	print((origH, origW))

	# img = Image.open(img_path)
	# img = np.array(img)[:,:,:3]
	# img = Image.fromarray(img)
	# draw = ImageDraw.Draw(img)
	# if boxes is None:
	# 	print("boxes is none") # 
	# 	return img

	results = []
	for box in boxes:# (bottom left,top right)
		padding =0.15
		startX = int(box[2])
		startY = int(box[3])
		endX = int(box[6])
		endY = int(box[7])

		dX = int((endX - startX) * padding)
		dY = int((endY - startY) * padding)

        # apply padding to each side of the bounding box, respectively
		startX = max(0, startX - dX)
		startY = max(0, startY - dY)
		endX = min(origW, endX + (dX * 2))
		endY = min(origH, endY + (dY * 2))

		if(startX>=endX or startY>=endY):
			continue

		# print(f"sx={startX},sy={startY},ex={endX},ey={endY}")
        # extract the actual padded ROI
		roi = orig[startY:endY, startX:endX]
		config = ("-l eng --oem 1 --psm 7")  # chi_sim
        # use Tesseract v4 to recognize a text ROI in an image
		text = pytesseract.image_to_string(roi, config=config)
		# strip out non-ASCII text so we can draw the text on the image using OpenCV
		text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
		text = text.replace('|', '')
		# print(f"ocr text = {text}")
		a = []
		a = re.findall("\d+\.?\d*", text) 
		# print(a)
		if(len(a)==0): continue
		text = int(float(a[0]))
		# # print(f"opencv text = {text}")
		results.append(((startX, startY, endX, endY), text))

	# sort the bounding boxes coordinates from top to bottom based on id(text)
	# results = sorted(results, key=lambda r:r[0][1])
	results = sorted(results, key=lambda r:r[1]) 
	output = orig.copy()
    # loop over the results
	print(len(results))
	for ((startX, startY, endX, endY), text) in results:
        # display the text OCR'd by Tesseract
		text = str(text)
		print(f"text ={text}")
        # draw the text and a bounding box surrounding the text region of the input image
		cv2.rectangle(output, (startX, startY), (endX, endY), (0, 0, 255), 2)
		cv2.putText(output, text, (startX, startY - 20),
			cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
	return output

def detect_dataset(model, device, test_img_path, submit_path):
	'''detection on whole dataset, save .txt results in submit_path
	Input:
		model        : detection model
		device       : gpu if gpu is available
		test_img_path: dataset path
		submit_path  : submit result for evaluation
	'''
	img_files = os.listdir(test_img_path)
	img_files = sorted([os.path.join(test_img_path, img_file) for img_file in img_files])
	
	for i, img_file in enumerate(img_files):
		print('evaluating {} image'.format(i), end='\r')
		img = Image.open(img_file).convert("RGB") 
		w, h = img.size
		boxes = detect(img, model, device) 
		# boxes = detect(Image.open(img_file), model, device)
		norm_boxes = boxes ## normalized boxes
		norm_boxes[:,[0,2,4,6]] = boxes[:,[0,2,4,6]]/w*1.0
		norm_boxes[:,[1,3,5,7]] = boxes[:,[1,3,5,7]] /h*1.0
		# norm_boxes[:,8] = text # id
		seq = []
		if boxes is not None:
			# seq.extend([','.join([str(b) for b in box[:-1]]) + '\n' for box in boxes]) 
			seq.extend([','.join([str(int(b)) for b in box[:-1]]) + '\n' for box in boxes]) 
		with open(os.path.join(submit_path, os.path.basename(img_file).replace('.png','.txt')), 'w') as f: 
			f.writelines(seq)

def detect_dataset_map(model, device, test_img_path, submit_path):
	'''detection on whole dataset, save .txt results in submit_path
	Input:
		model        : detection model
		test_img_path: dataset path
		submit_path  : submit result for evaluation
	'''
	img_files = os.listdir(test_img_path)
	img_files = sorted([os.path.join(test_img_path, img_file) for img_file in img_files])
	
	for i, img_file in enumerate(img_files): # img_file = img_path
		print('evaluating {} image'.format(i), end='\r')
		output_path = os.path.join(submit_path, os.path.basename(img_file).replace('.png','.pkl'))
		if  os.path.exists(output_path): 
			print("Already save1")
			continue 
		img = Image.open(img_file).convert("RGB")
		w, h = img.size
		boxes = detect(img, model, device) 

		image = cv2.imread(img_file) 
		orig = image.copy()
		origH, origW = image.shape[:2]
		padding =0.15 
		results = []
		lable_to_box = {} # Create a dictionary for a picture
		for box in boxes:
			
			startX = int(box[2]) # bottom left:x
			startY = int(box[3]) # bottom left:y
			endX = int(box[6]) # up right:x
			endY = int(box[7]) # up right:y

			dX = int((endX - startX) * padding)
			dY = int((endY - startY) * padding)

			# apply padding to each side of the bounding box, respectively
			startX = max(0, startX - dX)
			startY = max(0, startY - dY)
			endX = min(origW, endX + (dX * 2))
			endY = min(origH, endY + (dY * 2))

			if(startX>=endX or startY>=endY):
				continue # 
			# extract the actual padded ROI，and use Tesseract v4 to recognize a text ROI in an image
			roi = orig[startY:endY, startX:endX]
			config = ("-l eng --oem 1 --psm 7") 
			text = pytesseract.image_to_string(roi, config=config)
			## process text 
			# 1.Delete letters outside the ascii range
			text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
			text = text.replace('|', '')
			# 2.Get the number in text
			a = []
			a = re.findall("\d+\.?\d*", text)  
			if(len(a)==0): continue 
			text = int(float(a[0]))  
			## Convert to normalized coordinates and retain five decimal places
			startX  = np.around(startX/(w*1.0),5)
			endX  = np.around(endX/(w*1.0),5) 
			startY  = np.around(startY/(h*1.0),5)
			endY  = np.around(endY/(h*1.0),5) 
			lable_to_box[text] = [startX, startY, endX, endY] 
		# print(lable_to_box)
		if not os.path.exists(output_path):
			with open(output_path, 'wb') as f: 
				pickle.dump(lable_to_box,f) 
		else:
			print("Already save2")

if __name__ == '__main__':
	img_path    = './imgs/pos_ast/test/buggy/174388.png' 
	model_path  = './pths/model_epoch_190.pth' 
	res_img     = './res.bmp'  
	res_img1    = './res1.bmp' 
	img = Image.open(img_path)
	img = np.array(img)[:,:,:3]
	img = Image.fromarray(img)

	# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
	# model = EAST().to(device)
	# model.load_state_dict(torch.load(model_path))
	# model.eval()
	

	# print(len(boxes))
	# output=plot_boxes_text(img_path,boxes,res_img)
	# cv2.imwrite(res_img, output)

	# # # print(opencv_boxes)
	# plot_img = plot_boxes(img, boxes)	
	# plot_img.save(res_img1) 


