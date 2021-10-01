#! /usr/bin/env python3

import cv2
import numpy as np
import os

CHECKERBOARD = (5,7)
PATH = './images/set-2'

objpoints = []
imgpoints = [] 

objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.lib.index_tricks.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
print(objp)
prev_img_shape = None

images = os.listdir(PATH)
images = [i for i in images if i.startswith("chessboard_") and i.endswith('.jpg')]

for img_path in images:

	print(f"{PATH}/{img_path}")
	img = cv2.imread(f"{PATH}/{img_path}")
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
	
	if ret == True:
		objpoints.append(objp)
		imgpoints.append(corners)
		img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
	
	cv2.imshow('img',img)
	cv2.waitKey(0)

cv2.destroyAllWindows()

h,w = img.shape[:2]
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix:")
print(mtx)
print("Distortion Coefficients:")
print(dist)
print("rotation vectors:")
print(rvecs)
print("translation vectors:")
print(tvecs)

images = os.listdir(PATH)

for img_path in images:

	img = cv2.imread(f"{PATH}/{img_path}")
	h, w = img.shape[:2]
	newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
	undist = cv2.undistort(img, mtx, dist, None, newcameramtx)
	x, y, w, h = roi
	undist_cropped = undist[y:y+h, x:x+w]
	cv2.imshow('original', img)
	cv2.imshow('undistorted cropped', undist_cropped)
	cv2.waitKey(0) 