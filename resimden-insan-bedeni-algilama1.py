import cv2

img = cv2.imread("body.jpg")

body_cascade = cv2.CascadeClassifier("fullbody.xml")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)