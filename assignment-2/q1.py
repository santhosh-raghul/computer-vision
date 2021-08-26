import cv2
import sys
import numpy as np
import os

def sobel(img):
	_, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
	sob = cv2.Sobel(thresh, cv2.CV_8U, 1, 1)

	'''	
	# Monotonicity Transform
	sob = sob - np.min(sob)
	sob = sob/np.max(sob)
	sob *= 255
	'''
		
	return [np.count_nonzero(sob), sob.shape[0]*sob.shape[1], sob]

if __name__ == "__main__":

	if len(sys.argv)==2:
		img = cv2.imread(sys.argv[1], 0)

		edge_count, tot_count, sob = sobel(img)

		cv2.imshow(os.path.basename(sys.argv[1]), np.hstack([img, sob]))
		
		print(f"Number of edge pixels: {edge_count}")
		print(f"Totel number of pixels: {tot_count}")
		
		cv2.waitKey(0)
	else:
		print(f"{sys.argv[0]}: invalid usage\ncorrect usage: {sys.argv[0]} path_to_image",file=sys.stderr)