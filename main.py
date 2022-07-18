import numpy as np
import cv2 as cv
import imutils
import easyocr
import torch.cuda

torch.cuda.is_available()

img = cv.imread("./license-plates/Carro.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.imshow("img",  cv.cvtColor(gray,cv.COLOR_BGR2RGB))

bfilter = cv.bilateralFilter(gray, 11, 17, 17) #Noise reduction
edged = cv.Canny(bfilter, 30, 200) #Edge Detectiton
cv.imshow("Edge detection", cv.cvtColor(edged,cv.COLOR_BGR2RGB))

keypoints = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]

location = None

for contour in contours:
    approx = cv.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break

mask= np.zeros(gray.shape, np.uint8)
new_im= cv.drawContours(mask, [location], 0, 255, -1)
new_im = cv.bitwise_and(img, img, mask=mask)
cv.imshow("Segmented",cv.cvtColor(new_im, cv.COLOR_BGR2RGB) )

(x,y) = np.where(mask==255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1: y2+1]

cv.imshow("Cropped image",cv.cvtColor(cropped_image, cv.COLOR_BGR2RGB) )

reader = easyocr.Reader(["en"])
result = reader.readtext(cropped_image, allowlist = '-0123456789ABCDEFGHJKLMNPQRSTTUVWXYZ')
print(result)

text = result[0][-2]
font = cv.FONT_HERSHEY_SIMPLEX
res = cv.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+40), fontFace=font, fontScale=1, color=(0,255,255), thickness=2, lineType=cv.LINE_AA)
res = cv.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
cv.imshow("Result",cv.cvtColor(res, cv.COLOR_BGR2RGB))

cv.waitKey(0)
