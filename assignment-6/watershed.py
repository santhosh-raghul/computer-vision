import cv2
import numpy as np

img = cv2.imread("example1.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("img", img)

sobelx = cv2.Sobel(img,cv2.CV_64F,1,0) # Find x and y gradients
sobely = cv2.Sobel(img,cv2.CV_64F,0,1)


gX = cv2.convertScaleAbs(sobelx)
gY = cv2.convertScaleAbs(sobely)
combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)


# combine the gradient representations into a single image
combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)

# show our output images
cv2.imshow("Sobel/Scharr X", gX)
cv2.imshow("Sobel/Scharr Y", gY)
cv2.imshow("Sobel/Scharr Combined", combined)


# cv2.watershed(combined, )
# out = cv2.connectedComponents(img)
# print (out[1].shape)
# print (img.shape)

# cv2.imshow("conncomp", out[1])

_, img = cv2.threshold(img,0, 255, cv2.THRESH_OTSU)
dist_transform = cv2.distanceTransform(img,cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
print(np.unique(sure_fg))
print(np.unique(dist_transform))
cv2.imshow("sure_fg", sure_fg)

k_3x3 = np.ones((3,3), np.uint8)

def opencv_segmentation(mask, kernel=k_3x3, k=3):
	# noise removal
	opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=k)

	# sure background area
	sure_bg = cv2.dilate(opening, kernel, iterations=k)

	# Finding sure foreground area
	dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2, 5)
	ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

	# Finding unknown region
	sure_fg = np.uint8(sure_fg)
	unknown = cv2.subtract(sure_bg, sure_fg)

	# Marker labelling
	ret, markers = cv2.connectedComponents(sure_fg)

	# Add one to all labels so that sure background is not 0, but 1
	markers = markers + 1

	# Now, mark the region of unknown with zero
	markers[unknown > 0] = 0

	labels_ws = cv2.watershed(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB), markers)

	if labels_ws.max() - 1 < 2:
		return [mask], labels_ws

	res_masks = []
	for idx in range(2,  labels_ws.max() + 1):
		m = labels_ws == idx
		if m.sum() > 5:
			m = cv2.dilate(m.astype(np.uint8), kernel, iterations=1)
			res_masks.append(m)
	return res_masks, labels_ws 

masks,labels = opencv_segmentation(img)
print(masks,labels)

def watershed(rgb, idx, mask):
	'''
	Get watershed transform from image
	'''

	# kernel definition
	kernel = np.ones((3, 3), np.uint8)

	# sure background area
	sure_bg = cv2.dilate(mask, kernel)
	sure_bg = np.uint8(sure_bg)
	# util.im_gray_plt(sure_bg,"sure back")

	# Finding sure foreground area
	dist_transform = cv2.distanceTransform(np.uint8(mask), cv2.DIST_L2, 3)
	# util.im_gray_plt(dist_transform,"dist transform")
	ret, sure_fg = cv2.threshold(
		dist_transform, 0.5 * dist_transform.max(), 255, 0)

	# Finding unknown region
	sure_fg = np.uint8(sure_fg)
	# util.im_gray_plt(sure_fg,"sure fore")

	unknown = cv2.subtract(sure_bg, sure_fg)
	# util.im_gray_plt(unknown,"unknown")

	# marker labelling
	ret, markers = cv2.connectedComponents(sure_fg)

	# add one to all labels so that sure background is not 0, but 1
	markers = markers + 1

	# mark the region of unknown with zero
	markers[unknown == 255] = 0

	# util.im_gray_plt(np.uint8(markers),"markers")

	# apply watershed
	markers = cv2.watershed(rgb, markers)

	# create limit mask
	mask = np.zeros(mask.shape, np.uint8)
	mask[markers == -1] = 255

	return mask

mask = watershed(img, None, k_3x3)

cv2.imshow("disp",cv2.equalizeHist( (mask).astype(np.uint8) ))

cv2.waitKey(0)