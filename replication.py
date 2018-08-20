import cv2
import numpy as np
import math
import sys

IMG_PATH = sys.argv[1]
SCALE_X = float(sys.argv[2])
SCALE_Y = float(sys.argv[3])
RGB = int(sys.argv[4])	#1 for rgb and 0 for grayscale
IMG = cv2.imread(IMG_PATH, RGB)

def write_to_img(img, out_path):
	cv2.imwrite(img, out_path)
	
def scale_up():	#works for both rgb and grayscale
	im1 = np.repeat(IMG, int(SCALE_X), 1)	#repeat the cols.
	out_img = np.repeat(im1, int(SCALE_Y), 0)
	return out_img	
	
def scale_down():
	#delete every scale_X column and every scale_Y row
	L_col = []
	L_row = []
	stride_col = int(math.ceil(1.0/SCALE_X))
	stride_row = int(math.ceil(1.0/SCALE_Y))
	cols = IMG.shape[1]
	rows = IMG.shape[0]
	for i in range(1, cols, stride_col):
		for j in range(i, i + stride_col - 1):
			L_col.append(j)
			
	for i in range(1, rows, stride_row):
		for j in range(i, i + stride_row - 1):
			L_row.append(j)

	im1 = np.delete(IMG, L_col, 1)
	out_img = np.delete(im1, L_row, 0)
	return out_img 

#************************************************************************************

if(SCALE_X > 1 and SCALE_Y > 1):
	out = scale_up()
	write_to_img('new' + IMG_PATH, out)
else:
	out = scale_down()
	write_to_img('new' + IMG_PATH, out)
