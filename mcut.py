import cv2
import numpy as np
import sys
import dithering as dit
import edge_detection as ed

IMG_PATH = sys.argv[1]
IMG = cv2.imread(IMG_PATH, 1)

rows =len(IMG)
cols =len(IMG[0])

num_clr = int(sys.argv[2])
rdict = {}
bdict = {}
gdict = {}
coldict = {}

colors = []
K = 1
def gd(Tuple):
	return Tuple[1]
def rd(Tuple):
	return Tuple[2]
def bd(Tuple):
	return Tuple[0]
def returnColorMap(LIST):
	global K
	#print K
	if K == num_clr :
		return LIST
	newList = []
	for x in LIST:
		lt = x[0]
		bl = x[1][0]
		gl = x[1][1]
		rl = x[1][2]
		hl = int(len(lt)/2)
		if(bl >= gl and bl>= rl):
			sl = sorted(lt,key=bd)
			newList.append((sl[:hl],(bl/2,gl,rl)))
			newList.append((sl[hl:],(bl/2,gl,rl)))
		elif (gl >= bl and gl>= rl):
			sl = sorted(lt,key=gd)
			newList.append((sl[:hl],(bl,gl/2,rl)))
			newList.append((sl[hl:],(bl,gl/2,rl)))
		elif (rl >= bl and rl>= gl):
			sl = sorted(lt,key=rd)
			newList.append((sl[:hl],(bl,gl,rl/2)))
			newList.append((sl[hl:],(bl,gl,rl/2)))
	K*=2
	return returnColorMap(newList)


for i in range(rows):
	for j in range(cols):
		blue = IMG[i][j][0]
		green = IMG[i][j][1]
		red = IMG[i][j][2]
		try:
			bdict[blue]
		except KeyError:
			bdict[blue] = 1
		try:
			gdict[green]
		except KeyError:
			gdict[green] = 1
		try:
			rdict[red]
		except KeyError:
			rdict[red] = 1
		try:
			coldict[(blue,green,red)]
		except KeyError:
			coldict[(blue,green,red)] = 1

for key in coldict:
	colors.append(key)

finalList = returnColorMap([(colors,(len(bdict),len(gdict),len(rdict)))])

colorMap = []
for x in finalList:
	bsum = 0
	rsum = 0
	gsum = 0
	#print (len(x[0]))
	for y in x[0]:
		bsum+=y[0]
		gsum+=y[1]
		rsum+=y[2]
	colorMap.append((bsum/len(x[0]),gsum/len(x[0]),rsum/len(x[0])))

def getcolor(Tuple):
	minD = (colorMap[0][0] - Tuple[0])**2 + (colorMap[0][1] - Tuple[1])**2 + (colorMap[0][2] - Tuple[2])**2
	clr = (0,0,0,0)
	for x in colorMap:
		D = (x[0] - Tuple[0])**2 + (x[1] - Tuple[1])**2 + (x[2] - Tuple[2])**2
		if D<minD:
			minD = D
			clr = x
	return clr
new_img  = np.zeros(IMG.shape)

for i in range(rows):
	for j in range(cols):
		color = getcolor(IMG[i][j])
		new_img[i][j][0] = color[0]
		new_img[i][j][1] = color[1]
		new_img[i][j][2] = color[2]

new_img[0][1][0] = (new_img[0][1][0] - 1) % 255
new_img[0][1][1] = (new_img[0][1][1] - 1) % 255
new_img[0][1][2] = (new_img[0][1][2] - 1) % 255

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

#final_img = dit.floyd(IMG, colorMap, 1)

#if(np.array_equal(final_img, new_img)):
#	print('hello')

#cv2.imwrite("dithered"+IMG_PATH,final_img)
cv2.imwrite("mediancut"+IMG_PATH,new_img)

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
