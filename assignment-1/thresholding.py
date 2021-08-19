#!/usr/bin/env python3

import cv2, sys, numpy as np
from os.path import basename

thres_types = [ cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV, cv2.THRESH_TRUNC, cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV, cv2.THRESH_OTSU, cv2.THRESH_TRIANGLE ]
thres_type_names = ["binary", "binary inv", "trunc", "tozero", "tozero inv", "otsu", "triangle"]

def minmax(*images):
	l = []
	for image in images:
		l.extend([np.min(image),np.max(image)])
	return l

def threshold_with_trackbar(img, window_title="thresholding"):

	thres_type_label = "threshold type\n0 - binary\n1 - binary inverse\n2 - trunc\n3 - tozero\n4 - tozero inverse\n5 - otsu\n6 - triangle\n"
	img_thresh = None
	img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	img_gray_3 = cv2.cvtColor(img_gray,cv2.COLOR_GRAY2BGR)

	print("+============+=============+=======+=======+=======+=======+=======+=======+==========+==========+===========+===========+")
	print("| thres type | thres value | min r | max r | min g | max g | min b | max b | min gray | max gray | min thres | max thres |")
	print("+============+=============+=======+=======+=======+=======+=======+=======+==========+==========+===========+===========+")

	def update_img(val):

		thres_value = cv2.getTrackbarPos("threshold value", window_title)
		thres_type_index = cv2.getTrackbarPos(thres_type_label, window_title)
		thres_type = thres_types[thres_type_index]

		_,img_thresh = cv2.threshold(img_gray,thres_value,255,thres_type)
		img_thresh_3 = cv2.cvtColor(img_thresh,cv2.COLOR_GRAY2BGR)
		cv2.imshow(window_title, np.hstack((img,img_gray_3,img_thresh_3)))

		l = [thres_type_names[thres_type_index],thres_value]
		l.extend( minmax( img[:,:,2], img[:,:,1], img[:,:,0], img_gray, img_thresh ) )
		print("|{:^12}|{:^13}|{:^7}|{:^7}|{:^7}|{:^7}|{:^7}|{:^7}|{:^10}|{:^10}|{:^11}|{:^11}|".format(*l))

	cv2.namedWindow(window_title)
	cv2.createTrackbar(thres_type_label, window_title, 0, 6, update_img)
	cv2.createTrackbar("threshold value", window_title, 128, 255, update_img)
	update_img(128)
	cv2.waitKey()

	print("+============+=============+=======+=======+=======+=======+=======+=======+==========+==========+===========+===========+")
	return img_thresh

if __name__=="__main__":

	if len(sys.argv)==2:
		threshold_with_trackbar(cv2.imread(sys.argv[1]),basename(sys.argv[1]))

	else:
		print(f"{sys.argv[0]}: invalid usage\ncorrect usage: {sys.argv[0]} path_to_image",file=sys.stderr)