import cv2
import sys
import math
import numpy as np
img = cv2.imread(sys.argv[1],int(sys.argv[2]))
x = len(img[0])
X = int(x*float(sys.argv[3]))
y = len(img)
Y = int(y*float(sys.argv[4]))
print (x,y)
if(sys.argv[2]=='0'):
	new_img = np.zeros((Y,X))
else:
	new_img = np.zeros((Y,X,3))
x_spc = float(x-1)/float(X-1)
y_spc = float(y-1)/float(Y-1)
def getx(p,q):
	#print(p,q)
	if sys.argv[2] == '0':
		if p.is_integer() and q.is_integer():
			return img[int(y-q-1)][int(p)]
		elif p.is_integer():
			y1 = math.floor(q)
			y2 = math.ceil(q)
			return int(((y2-q)*img[int(y-q-1)][int(p)]+(q-y1)*img[int(y-q-1)][int(p)])/(y2-y1))
		elif q.is_integer():
			x1 = math.floor(p)
			x2 = math.ceil(p)
			return int(((x2-p)*img[int(y-q-1)][int(x1)]+(p-x1)*img[int(y-q-1)][int(x2)])/(x2-x1))
		else:
			x1 = math.floor(p)
			x2 = math.ceil(p)
			y1 = math.floor(q)
			y2 = math.ceil(q)
			A1 = np.array([[x2-p,p-x1]])
			A2 = np.array([[img[int(y-1-y1)][int(x1)],img[int(y-y2-1)][int(x1)]],[img[int(y-y1-1)][int(x2)],img[int(y-y2-1)][int(x2)]]])
			A3 = np.array([y2-q,q-y1])
			return int(np.matmul(np.matmul(A1,A2b),A3)[0]/((x2-x1)*(y2-y1)))
	else:
		if p.is_integer() and q.is_integer():
			return img[int(y-q-1)][int(p)]
		elif p.is_integer():
			y1 = math.floor(q)
			y2 = math.ceil(q)
			return (((y2-q)*img[int(y-q-1)][int(p)]+(q-y1)*img[int(y-q-1)][int(p)])/(y2-y1))
		elif q.is_integer():
			x1 = math.floor(p)
			x2 = math.ceil(p)
			return (((x2-p)*img[int(y-q-1)][int(x1)]+(p-x1)*img[int(y-q-1)][int(x2)])/(x2-x1))
		else:
			x1 = math.floor(p)
			x2 = math.ceil(p)
			y1 = math.floor(q)
			y2 = math.ceil(q)
			A1 = np.array([[x2-p,p-x1]])
			A2b = np.array([[img[int(y-1-y1)][int(x1)][0],img[int(y-y2-1)][int(x1)][0]],[img[int(y-y1-1)][int(x2)][0],img[int(y-y2-1)][int(x2)][0]]])
			A2g = np.array([[img[int(y-1-y1)][int(x1)][1],img[int(y-y2-1)][int(x1)][1]],[img[int(y-y1-1)][int(x2)][1],img[int(y-y2-1)][int(x2)][1]]])
			A2r = np.array([[img[int(y-1-y1)][int(x1)][2],img[int(y-y2-1)][int(x1)][2]],[img[int(y-y1-1)][int(x2)][2],img[int(y-y2-1)][int(x2)][2]]])
			A3 = np.array([y2-q,q-y1])
			return np.array([int(np.matmul(np.matmul(A1,A2b),A3)[0]/((x2-x1)*(y2-y1))),int(np.matmul(np.matmul(A1,A2g),A3)[0]/((x2-x1)*(y2-y1))),int(np.matmul(np.matmul(A1,A2r),A3)[0]/((x2-x1)*(y2-y1)))])

for i in range(Y):
	#print(i)
	for j in range(X):
		new_img[Y-i-1][j] = getx(x_spc*j,y_spc*i)

cv2.imwrite('new'+sys.argv[1],new_img)

#spacing = x1-1/	