import cv2
import numpy as np
import sys

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

cv2.imwrite("painted"+IMG_PATH,new_img)

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

