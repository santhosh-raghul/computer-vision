import cv2
import numpy as np

def monotonicity(img):
	img -= np.min(img)
	img //= np.max(img)
	img *= 255
	return img.astype(np.uint8)

img_orig = cv2.imread("../images/chessboard.png")
img_orig = cv2.imread("../images/IMG_20210902_232538__02.jpg")
scale = 8
img_orig = cv2.resize(img_orig, (img_orig.shape[1]//scale, img_orig.shape[0]//scale))
img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

cv2.imshow("thresh", thresh)
cv2.waitKey(0)

_, hori = cv2.threshold(monotonicity(np.abs(cv2.Sobel(thresh, cv2.CV_16S, 1, 0))), 10, 255, cv2.THRESH_OTSU)
_, vert = cv2.threshold(monotonicity(np.abs(cv2.Sobel(thresh, cv2.CV_16S, 0, 1))), 10, 255, cv2.THRESH_OTSU)

hori_rect_e = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
hori_rect_d = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))

vert_rect_e = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
vert_rect_d = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))

cv2.imshow("hori_old", hori)
cv2.imshow("vert_old", vert)

hori = cv2.erode(hori, hori_rect_e)
hori = cv2.dilate(hori, hori_rect_d)

vert = cv2.erode(vert, vert_rect_e)
vert = cv2.dilate(vert, vert_rect_d)

cv2.imshow("hori", hori)
cv2.imshow("vert", vert)
cv2.imshow("and", vert & hori)
cv2.imshow("or", vert | hori)

cv2.waitKey(0)

conts = []
cnts, heir = cv2.findContours(vert & hori, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
for cnt in cnts:
	c, r = cv2.minEnclosingCircle(cnt)
	cv2.circle(img_orig, (round(c[0]), round(c[1])), 5, (0, 255, 0), 2)
	conts.append((round(c[0]), round(c[1])))

cv2.imshow("cnts", img_orig)

conts = sorted(conts, key=lambda x: x[0])

x = 0; y = 0
error = 15
all_lines = []

while(conts != []):
	curr = conts[0]
	curr_line = []
	for i in conts:
		if (curr[0] - error) <= i[0] <= (curr[0] + error):
			curr_line.append(i)

	for i in curr_line:
		conts.remove(i)

	curr_line = sorted(curr_line, key=lambda x: x[1])

	all_lines.append(curr_line)

for x, line in enumerate(all_lines):
	for y, point in enumerate(line):
		cv2.putText(img_orig, f'({x},{y})', point, cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255), 1)

cv2.namedWindow("final", cv2.WND_PROP_FULLSCREEN | cv2.WINDOW_NORMAL)# | cv2.WINDOW_AUTOSIZE)
cv2.imshow("final", img_orig)
cv2.waitKey(0)

'''
diag = cv2.Sobel(img, cv2.CV_8U, 1, 1)
diag_90 = cv2.Sobel(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), cv2.CV_8U, 1, 1)

diag_ = cv2.rotate(diag_90, cv2.ROTATE_90_COUNTERCLOCKWISE)

comb = diag | diag_
comb = monotonicity(np.abs(cv2.Sobel(img, cv2.CV_16S, 1, 1)))

cnts, heir = cv2.findContours(comb, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# print(cnts)
# cv2.drawContours(img_orig, cnts, -1, (0, 255, 0), 2)
print(len(cnts))
for cnt in cnts:
	c, r = cv2.minEnclosingCircle(cnt)
	cv2.circle(img_orig, (round(c[0]), round(c[1])), 5, (0, 255, 0), 2)
	# print((round(c[0]), round(c[1])))

print(len(cnts))

cv2.imshow("final", img_orig)
cv2.imshow("diag", diag)
cv2.imshow("diag_90", diag_)
cv2.imshow("comb", comb)

cv2.waitKey(0)
'''


'''
_, hori = cv2.threshold(cv2.Sobel(img, cv2.CV_8U, 1, 0), 10, 255, cv2.THRESH_BINARY)
_, vert = cv2.threshold(cv2.Sobel(img, cv2.CV_8U, 0, 1), 10, 255, cv2.THRESH_BINARY)

print(np.unique(vert & hori))

cv2.imshow("hori", hori)
cv2.imshow("vert", vert)
cv2.imshow("and", vert & hori)

cv2.waitKey(0)
'''