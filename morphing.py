import cv2
import imageio
import sys
import math
import numpy as np
import feature_points as fp
from scipy.spatial.qhull import Delaunay
'''
def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True
'''

def get_triangulation(size, feature_points):
	#print 'F', feature_points
	'''
	rect = (0, 0, size[1], size[0])
	subdiv = cv2.Subdiv2D(rect)
	for point in feature_points:
		subdiv.insert(point)
	triangleList = subdiv.getTriangleList()
	finalList = []
	for t in triangleList:		
		pt1 = (t[0], t[1])
		pt2 = (t[2], t[3])
		pt3 = (t[4], t[5])
		if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):
			finalList.append([t[0], t[1], t[2], t[3], t[4], t[5]])
	finalList = np.array(finalList)
	finalList = finalList.astype(int)
	return np.array(finalList)
	'''
	f_points = np.array(feature_points)
	tri = Delaunay(f_points)
	triangleList = f_points[tri.simplices]
	return triangleList
	
def get_cartesian_from_barycentric(p1, p2, p3, b):
	'''
	b are barycentric coordinates and p1, p2, p3 are vertices of the triangle
	'''
	x = b[0]*p1[0] + b[1]*p2[0] + b[2]*p3[0]
	y = b[0]*p1[1] + b[1]*p2[1] + b[2]*p3[1]
	return (int(x), int(y))	
	
def get_barycentric_from_cartesian(p1, p2, p3, p):
	'''
	p1, p2, p3 are the vertices of the triangle containing p and we wish to know barycentric coordinates of p
	these are: a, b,c such that p = a.p1 + b.p2 + c.p3 and a + b + c = 1
	'''
	#print p, 'PPPP\n'
	matrix = np.array([[p1[0], p2[0], p3[0]], [p1[1], p2[1], p3[1]], [1, 1, 1]])
	another = np.array([[p[0]], [p[1]], [1]])
	#print matrix, '\n', another
	inverse = np.linalg.inv(matrix)
	final = np.matmul(inverse, another)
	bary_coord = [final[0, 0], final[1, 0], final[2, 0]]
	return bary_coord
	
def check_inside_triangle(p1, p2, p3, p):
	'''
	find whether p lies inside/on triangle formed by p1, p2 and p3
	returns true if point is inside/on triangle, otherwise false
	'''
	l = [p1, p2, p3]
	ans = cv2.pointPolygonTest(np.array(l), p, measureDist = False)
	if ans == 1 or ans == 0:
		return True
	else:
		return False

def get_corresponding_triangle(T_im, im_points, source_points):
	p1 = T_im[0]
	p2 = T_im[1]
	p3 = T_im[2]
	ind_p1 = im_points.index(p1)
	ind_p2 = im_points.index(p2)
	ind_p3 = im_points.index(p3)
	
	T_corr = []
	T_corr.append(source_points[ind_p1])
	T_corr.append(source_points[ind_p2])
	T_corr.append(source_points[ind_p3])
	return T_corr

def get_intermediate_images(src_image, dest_image, source_points, destination_points, source_triangulation, destination_triangulation, alpha):
	'''
	Generate the intermediate image for a given alpha(interpolation) parameter
	first obtain intermediate points by interpolating corresponding source and destination points; i.e. ik = (1 - alpha)mk + alphank
	then for each pixel in the intermediate image, find the triangle in which it lies and then its barycentric coordinates
	then find the corresponding triangle in the source and destination image
	then find the cartesian coordinate in the source and destination image which has same barycentric coordinates	
	'''	
	im_points = []
	size = dest_image.shape
	im_image = np.zeros(size)
	
	for pair in zip(source_points[:-4], destination_points[:-4]):
		#generate intermediate point
		point = (int((1-alpha)*pair[0][0] + alpha*pair[1][0]), int((1-alpha)*pair[0][1] + alpha*pair[1][1]))
		im_points.append(point)
	im_points = im_points + [(0, 0), (dest_image.shape[1] - 1, 0), (0, dest_image.shape[0] - 1), (dest_image.shape[1] - 1, dest_image.shape[0] - 1)]	
	#obtain triangulation of im_points now
	im_triangulation = get_triangulation(size, im_points)
	#print 'IM: ', '\n', im_triangulation, '\n'
	
	for i in range(size[0]):
		for j in range(size[1]):
			T_im = []
			print 'i, j:', (i, j), '\n'
			for t in im_triangulation:
				p1 = (t[0][0], t[0][1])
				p2 = (t[1][0], t[1][1])
				p3 = (t[2][0], t[2][1])
				p = (j, i)
				if check_inside_triangle(p1, p2, p3, p):
					T_im.append(p1)
					T_im.append(p2)
					T_im.append(p3)
					break
			
			T_src = get_corresponding_triangle(T_im, im_points, source_points)
			T_dest = get_corresponding_triangle(T_im, im_points, destination_points)
			
			#get barycentric coordinates of (i, j) now relative to T_im
			bary_coord = get_barycentric_from_cartesian(T_im[0], T_im[1], T_im[2], (j, i))
			#now use same bary_coord in T_src and T_dest to get corresponding cartesian coordinates 
			src_x, src_y = get_cartesian_from_barycentric(T_src[0], T_src[1], T_src[2], bary_coord)
			dest_x, dest_y = get_cartesian_from_barycentric(T_dest[0], T_dest[1], T_dest[2], bary_coord)
			#now get colour of (i, j) by interpolating src_image[src_x][src_y] and dest_image[dest_x][dest_y]
			im_image[i][j][0] = int((1-alpha)*src_image[src_y][src_x][0] + alpha*dest_image[dest_y][dest_x][0])	
			im_image[i][j][1] = int((1-alpha)*src_image[src_y][src_x][1] + alpha*dest_image[dest_y][dest_x][1])	
			im_image[i][j][2] = int((1-alpha)*src_image[src_y][src_x][2] + alpha*dest_image[dest_y][dest_x][2])	
	
	return im_image		

def generate_gif(list_images):
	images = []
	for filename in list_images:
		images.append(imageio.imread(filename))
	output_file = 'new.gif'
	imageio.mimsave(output_file, images, duration=0.3)

if __name__ == "__main__":
	'''
	Usage: python morphing.py --source_img path --destination_img path num_intermediate_images
	'''
	print 'Start selecting feature points on source and destination images, NOTE: select points alternately on source and destination and when done enter c!\n'
	src_image = cv2.imread(sys.argv[1])
	dest_image = cv2.imread(sys.argv[2])
	num_images = int(sys.argv[3]) + 1
	
	if src_image.shape != dest_image.shape:
		src_image = cv2.resize(src_image, (dest_image.shape[1], dest_image.shape[0]))
	
	source_points, destination_points = fp.get_feature_points(sys.argv[1], sys.argv[2])
	if len(source_points) != len(destination_points):
		print 'You have selected unequal number of points!\n'
		sys.exit(0)
	
	source_points = source_points + [(0, 0), (dest_image.shape[1] - 1, 0), (0, dest_image.shape[0] - 1), (dest_image.shape[1] - 1, dest_image.shape[0] - 1)]
	destination_points = destination_points + [(0, 0), (dest_image.shape[1] - 1, 0), (0, dest_image.shape[0] - 1), (dest_image.shape[1] - 1, dest_image.shape[0] - 1)]
	
	#print source_points, '\n', destination_points, '\n', src_image.shape,'\n', dest_image.shape	
	source_triangulation = get_triangulation(dest_image.shape, source_points)
	destination_triangulation = get_triangulation(dest_image.shape, destination_points)
	
	list_images = [src_image]
	it = np.linspace(0, 1, num_images, endpoint = False)
	for alpha in it:
		if(alpha == 0):
			continue
		print 'Alpha: ', alpha, ' done!\n\n'
		img = get_intermediate_images(src_image, dest_image, source_points, destination_points, source_triangulation, destination_triangulation, alpha)
		#print '\n\n HELLO \n\n'
		list_images.append(img)
		#print '\n\n Image ', alpha, ' done\n\n'
	
	list_images.append(dest_image)
	name_images = [sys.argv[1]]
	counter = 1
	for img in list_images[1:-1]:
		name = 'im' + str(counter) + '.jpeg'
		counter += 1
		cv2.imwrite(name, img)
		name_images.append(name)
	
	name_images.append(sys.argv[2])
	print 'ALL DONE!\n'
	if sys.argv[4] == '1':
		generate_gif(name_images)
	
