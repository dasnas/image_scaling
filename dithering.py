import cv2
import numpy as np
import math
import sys

IMG_PATH = sys.argv[1]
IMG = cv2.imread(IMG_PATH, 1)

def write_to_img(img, out_path):
	cv2.imwrite(img, out_path)
	
def getcolor(Tuple, colorMap):
	minD = (colorMap[0][0] - Tuple[0])**2 + (colorMap[0][1] - Tuple[1])**2 + (colorMap[0][2] - Tuple[2])**2
	clr = (0,0,0)
	for x in colorMap:
		D = (x[0] - Tuple[0])**2 + (x[1] - Tuple[1])**2 + (x[2] - Tuple[2])**2
		if D<minD:
			minD = D
			clr = x
	return clr
	
def floyd(input_img, colorMap):
	out_img = np.copy(input_img)
	rows = out_img.shape[0]
	cols = out_img.shape[1]
	
	for i in range(rows - 1):
		for j in range(1, cols - 1):
			old_b = out_img[i][j][0]
			old_g = out_img[i][j][1]
			old_r = out_img[i][j][2]
			new = getcolor(out_img[i][j], colorMap)
			
			out_img[i][j][0] = new[0]
			out_img[i][j][1] = new[1]
			out_img[i][j][2] = new[2]
			
			b_quant_e = old_b - new[0]
			g_quant_e = old_g - new[1]
			r_quant_e = old_r - new[2]
		
			out_img[i][j + 1][0] += round(b_quant_e * 7/16)
			out_img[i][j + 1][1] += round(g_quant_e * 7/16)
			out_img[i][j + 1][2] += round(r_quant_e * 7/16)

			out_img[i + 1][j - 1][0] += round(b_quant_e * 3/16)
			out_img[i + 1][j - 1][1] += round(g_quant_e * 3/16)
			out_img[i + 1][j - 1][2] += round(r_quant_e * 3/16)			
			
			out_img[i + 1][j][0] += round(b_quant_e * 5/16)
			out_img[i + 1][j][1] += round(g_quant_e * 5/16)
			out_img[i + 1][j][2] += round(r_quant_e * 5/16)
			
			out_img[i + 1][j + 1][0] += round(b_quant_e * 1/16)
			out_img[i + 1][j + 1][1] += round(g_quant_e * 1/16)
			out_img[i + 1][j + 1][2] += round(r_quant_e * 1/16)
	
	return out_img
