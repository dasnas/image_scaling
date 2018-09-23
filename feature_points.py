import cv2
import math

def click1(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONDOWN:
		param[0].append((x,y))
		cv2.line(param[1], (x, y), (x, y), (255,0,0), 2)
		
def click2(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONDOWN:
		param[0].append((x,y))
		cv2.line(param[1], (x, y), (x, y), (0,0,255), 2)

def get_feature_points(source_img, destination_img):
	'''
	source_img path and destination_img path; both windows will open, user will mark points and two lists of same sizes will be returned
	'''
	src_image = cv2.imread(source_img)
	dest_image = cv2.imread(destination_img)
	
	dest_shape = dest_image.shape	
	
	if src_image.shape != dest_shape:
		src_image = cv2.resize(src_image, (dest_image.shape[1], dest_image.shape[0]))
	
	src_shape = src_image.shape		
	source_points = []
	destination_points = []
		
	cv2.namedWindow("src_image")
	cv2.setMouseCallback("src_image", click1, (source_points, src_image))
	cv2.namedWindow("dest_image")
	cv2.setMouseCallback("dest_image", click2, (destination_points, dest_image))
	
	while True:
		cv2.imshow("src_image", src_image)
		key = cv2.waitKey(1) & 0xFF
		
		cv2.imshow("dest_image", dest_image)
		key = cv2.waitKey(1) & 0xFF
		
		if key == ord("c"):
			break

	cv2.destroyAllWindows()
	return (source_points, destination_points)
