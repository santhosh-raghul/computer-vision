# import the necessary packages
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import argparse
import imutils
import colorsys
import cv2

def monotonicity_transform(img):

	img=(img-np.min(img)).astype(np.float64)
	img=img/np.max(img)
	img=(img*255).astype(np.uint8)
	return img

image = cv2.imread("example3.png")
shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)

cv2.imshow("Input", image)
cv2.imshow("shifted", shifted)

gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_OTSU)[1]
cv2.imshow("Thresh", thresh)

distTransImg = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
localMax = peak_local_max(distTransImg, indices=False, min_distance=20,	labels=thresh)
cv2.imshow("Distance Transform", monotonicity_transform(cv2.convertScaleAbs(distTransImg)))

markers = ndimage.label(localMax)[0]
labels = watershed(-distTransImg, markers, mask=thresh)
labels_count = len(np.unique(labels))

label_img = image.copy()

for q, label in enumerate(np.unique(labels)):
	mask = np.zeros(gray.shape, dtype="uint8")
	mask[labels == label] = 255

	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	color = colorsys.hsv_to_rgb(q/labels_count*360, 1, 1)
	color = [i*255 for i in color]
	print(label, color)

	cv2.drawContours(label_img, cnts[0], -1, color, -1)

cv2.imshow("Segmented Image", label_img)
cv2.imshow("Output", monotonicity_transform(0.7*image + 0.3*label_img))
cv2.waitKey(0)