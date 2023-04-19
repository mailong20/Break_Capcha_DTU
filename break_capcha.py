import cv2
import pytesseract
import numpy as np
tessdata_dir_config = '--tessdata-dir "C:\\Program Files\\Tesseract-OCR\\tessdata"'
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

img = cv2.imread(r'E:\PROJECT\capcha\capcha_mydtu\image_capcha\00ee2c3b-1fd8-4b4c-97ff-a4d2a0807d6c.png')
img = cv2.resize(img, None, fx=2, fy=2)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
adaptive = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 20)
contours, hierarchy = cv2.findContours(255-adaptive, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
cnt = sorted(contours, key=cv2.contourArea, reverse=True)[:4]
mask = 255 - np.zeros_like(adaptive)
cv2.drawContours(mask, cnt, -1, 0, -1)
adaptive[np.where(mask!=0)]=255
images = [cv2.boundingRect(cnt) for cnt in contours]
images = sorted(images, key=lambda x:x[0])
images = [adaptive[x:x+w, y:y+h] for x, y, w, h in images]
print((pytesseract.image_to_string(images[0])).strip())

cv2.imshow("Captcha", mask) # Output: IMQW
cv2.imshow("Gray", gray) # Output: IMOW
cv2.imshow("Adaptive", adaptive) # Output: IMOW,

cv2.waitKey()