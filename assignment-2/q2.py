import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.shape_base import hstack
import q1

def gauss(img):
	sizes = [(7, 7), (9, 9), (11, 11)]
	out = []
	for size in sizes:
		gauss_img = cv2.GaussianBlur(img, size, 0)
		edge_count, tot_count, sob = q1.sobel(gauss_img)
		print(f"----------{size}----------")
		print(f"Number of edge pixels: {edge_count}")
		print(f"Totel number of pixels: {tot_count}")
		
		out.append(np.vstack([gauss_img, sob]))

	cv2.imshow(f"output", np.hstack(out))
	
	cv2.waitKey(0)

if __name__ == "__main__":

	if len(sys.argv)==2:
		gauss(cv2.imread(sys.argv[1], 0))

	else:
		print(f"{sys.argv[0]}: invalid usage\ncorrect usage: {sys.argv[0]} path_to_image",file=sys.stderr)