import cv2
import numpy as np
import sys
import edge_detection as ed
import dithering as dit

IMG_PATH = sys.argv[1]
IMG = cv2.imread(IMG_PATH, 1)

histogram = np.zeros((32,32,32))
b = np.zeros((32,32,32))
g = np.zeros((32,32,32))
r = np.zeros((32,32,32))


def comp(Tuple):
	return Tuple[0]

num_clr = int(sys.argv[2])
rows =len(IMG)
cols =len(IMG[0])

for i in range(rows):
	for j in range(cols):
		cb = IMG[i][j][0]
		cg = IMG[i][j][1]
		cr = IMG[i][j][2]
		p = int(cb/8)
		q = int(cg/8)
		s = int(cr/8)
		histogram[int(p)][int(q)][int(s)]+=1
		b[p][q][s] += cb
		g[p][q][s] += cg
		r[p][q][s] += cr

color_map = []
pixels = []
for i in range(32):
	for j in range(32):
		for k in range(32):
			freq = histogram[i][j][k]
			if freq != 0:
				pixels.append((freq,b[i][j][k]/freq,g[i][j][k]/freq,r[i][j][k]/freq))
			else:
				pixels.append((0,0,0,0))
new_pixels = sorted(pixels,key=comp)
for i in range(32*32*32-1,32*32*32-num_clr-1,-1):
	color_map.append(new_pixels[i])
def getcolor(Tuple):
	minD = (color_map[0][1] - Tuple[0])**2 + (color_map[0][2] - Tuple[1])**2 + (color_map[0][3] - Tuple[2])**2
	clr = (0,0,0,0)
	for x in color_map:
		D = (x[1] - Tuple[0])**2 + (x[2] - Tuple[1])**2 + (x[3] - Tuple[2])**2
		if D<minD:
			minD = D
			clr = x
	return clr
new_img  = np.zeros(IMG.shape)

for i in range(rows):
	for j in range(cols):
		color = getcolor(IMG[i][j])
		new_img[i][j][0] = color[1]
		new_img[i][j][1] = color[2]
		new_img[i][j][2] = color[3]

#******************************-------------------------****************************
def rendering(input_img_grayscale, color_q_img, ksize, sigma, tau, epsilon, pheta, threshold):
	#for painterly rendered image
	#in xdog image, edges are white and background is black
	out_img = ed.xDoG(input_img_grayscale, ksize, sigma, tau, epsilon, pheta)
	rows = input_img_grayscale.shape[0]
	cols = input_img_grayscale.shape[1]
	final_img = np.zeros((rows, cols, 3))
	for r in range(rows):
		for c in range(cols):
			if out_img[r][c] < threshold:	#background
				final_img[r][c][0] = color_q_img[r][c][0]
				final_img[r][c][1] = color_q_img[r][c][1]
				final_img[r][c][2] = color_q_img[r][c][2]
			else:
				final_img[r][c][0] = 0
				final_img[r][c][1] = 0
				final_img[r][c][2] = 0
				
	return final_img
#******************************----------------------------*************************

def get_threshold(input_grayscale_img, ksize, sigma, tau, epsilon, pheta):
	edge_img = ed.xDoG(input_grayscale_img, ksize, sigma, tau, epsilon, pheta)
	avg = 0
	rows = edge_img.shape[0]
	cols = edge_img.shape[1]
	for r in range(rows):
		for c in range(cols):
			avg += edge_img[r][c]
	avg = int(avg/(rows * cols))
	return avg

#*****************************----------------------------**************************

#final_img = dit.floyd(IMG, color_map, 0)
#cv2.imwrite("dithered"+IMG_PATH,final_img)
cv2.imwrite("pop"+IMG_PATH,new_img)
if (len(sys.argv) > 3):
	grayscale_img = cv2.imread(IMG_PATH, 0)
	ksize = int(sys.argv[3])
	sigma = float(sys.argv[4])
	tau = float(sys.argv[5])
	epsilon = float(sys.argv[6])
	pheta = float(sys.argv[7])
	
	if(len(sys.argv) > 8):
		threshold = int(sys.argv[8])
	else:
		threshold = get_threshold(grayscale_img, ksize, sigma, tau, epsilon, pheta)
	
	out_painted = rendering(grayscale_img, new_img, ksize, sigma, tau, epsilon, pheta, threshold)
	cv2.imwrite('out_painted'+IMG_PATH, out_painted)

'''
hist = cv2.calcHist([IMG],[0,1,2],None,[cr,cr,cr],[0,256,0,256,0,256])
pixels = []
for i in range(cr):
	for j in range(cr):
		for k in range(cr):
			pixels.append((hist[i][j][k],i*8+4,j*8+4,k*8+4))
new_pixels = sorted(pixels,key=comp)
color_map = []
-----------------------
incorrect :
for i in range(cr*cr*cr-1,cr*cr*cr-cr-1,-1):
correct :
for i in range(32*32*32-1,32*32*32-cr-1,-1):
	----------------------------------
	color_map.append(new_pixels[i])
def getcolor(Tuple):
	minD = (color_map[0][1] - Tuple[0])**2 + (color_map[0][2] - Tuple[1])**2 + (color_map[0][3] - Tuple[2])**2
	clr = (0,0,0,0)
	for x in color_map:
		D = (x[1] - Tuple[0])**2 + (x[2] - Tuple[1])**2 + (x[3] - Tuple[2])**2
		if D<minD:
			minD = D
			clr = x
	return clr
new_img  = np.zeros(IMG.shape)

for i in range(rows):
	for j in range(cols):
		color = getcolor(IMG[i][j])
		new_img[i][j][0] = color[1]
		new_img[i][j][1] = color[2]
		new_img[i][j][2] = color[3]

cv2.imwrite("painted"+IMG_PATH,new_img)
'''

