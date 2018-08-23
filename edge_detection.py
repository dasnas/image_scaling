import numpy as np
import math
import cv2
import sys
'''
def kernel(size, sigma):
	#size is odd
	loop_indi = size - 1
	loop_indj = 0
	su = 0
	mask = np.zeros((size, size))
	for i in range(-1 * size/2 + 1, 1 * size/2 + 1):
		loop_indi = size - 1
		for j in range(-1 * size/2 + 1, 1 * size/2 +1 ):
			#print i, j, '\n'
			mask[loop_indi][loop_indj] = (math.exp(1.0 * (-1 * (i**2 + j**2))/ (2.0 * sigma**2)))
			su += mask[loop_indi][loop_indj]
			loop_indi -= 1
		loop_indj += 1
	
	#mask = np.multiply(mask, 1/su)
	return mask

def convolve(img, kernel):
	rows = img.shape[0]
	cols = img.shape[1]
	ksize = kernel.shape[0]
	out_img = np.copy(img)
	for i in range(rows):
		for j in range(cols):
			if (i < ksize/2 or rows - i - 1< ksize/2 or j < ksize/2 or cols - j - 1 < ksize/2):
				continue
			else:
				#print i - ksize/2, ' ',  i + ksize/2 + 1,' ',  j - ksize/2, ' ',  j + ksize/2+1, '\n'
				submatrix = img[i - ksize/2: i + ksize/2 + 1, j - ksize/2: (j + ksize/2+1)]
				out = np.multiply(submatrix, kernel)
				res = out.sum()
				out_img[i][j] = res	
	return out_img
	
filt1 = kernel(3,1)
#print filt
filt2 = kernel(3, 1.6)
img = cv2.imread('apj.jpg', 0)
print img, '\n'
out1 = convolve(img, filt1)
out2 = convolve(img, filt2)
out = out1 - out2
print out
'''

'''
optimal params:
ksize: 5
sigma: 1.2
tau: 0.984
epsilon: ~0?
pheta: ~1?
'''

def xDoG(img, ksize, sigma, tau, epsilon, pheta):	
	out1 = cv2.GaussianBlur(img, (ksize,ksize), sigma, 0)
	out2 = cv2.GaussianBlur(img, (ksize,ksize), 1.6*sigma, 0)
	dog = out1 - tau * out2
	dog = dog.astype(np.uint8)
	rows = img.shape[0]
	cols = img.shape[1]
	xdog = np.zeros((rows, cols))
	for i in range(rows):
		for j in range(cols):
			if dog[i][j] < epsilon:
				xdog[i][j] = 1
			else:				
				xdog[i][j] = int(round(1 + math.tanh(1.0 * pheta * dog[i][j])))
	#return xdog
	xdog = xdog.astype(np.uint8)
	return dog
	
if __name__ == "__main__":
	img = cv2.imread(sys.argv[1], 0)
	ksize = int(sys.argv[2])
	sigma = float(sys.argv[3])
	tau = float(sys.argv[4])
	epsilon = float(sys.argv[5])
	pheta = float(sys.argv[6])
	Final = xDoG(img, ksize, sigma, tau, epsilon, pheta)
	cv2.imwrite('edges' + sys.argv[1], Final)

