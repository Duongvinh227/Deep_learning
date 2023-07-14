import cv2

img_label1 = cv2.imread("demo_detec_3.png")

# img_detec_1 = img_label1[43:54, 48:216]
# img_detec_2 = img_label1[31:41, 62:195]
# img_detec_3 = img_label1[22:32, 48:208]
# img_detec_4 = img_label1[11:23, 64:193]
# img_detec_5 = img_label1[23:35, 227:296]
# img_detec_6 = img_label1[35:46, 226:300]
# img_detec_7 = img_label1[5:17, 4:37]
# 43 54 48 216
# cv2.imwrite("demo_detec_1.png", img_detec_1)
#31 41 62 195
# cv2.imwrite("demo_detec_22.png", img_detec_2)
#22 32 48 208
# cv2.imwrite("demo_detec_3.png", img_detec_3)
#11 23 64 193
# cv2.imwrite("demo_detec_4.png", img_detec_4)
#23 35 227 296
# cv2.imwrite("demo_detec_5.png", img_detec_5)
#36 45 226 300
# cv2.imwrite("demo_detec_6.png", img_detec_6)
# 5 17 4 37
# cv2.imwrite("demo_detec_7.png", img_detec_7)

gray_image = cv2.cvtColor(img_label1, cv2.COLOR_RGB2GRAY)
gray_image = cv2.resize(gray_image,(300,32))
cv2.imwrite("demo_detec_3_gray.png", gray_image)

cv2.imshow("name", gray_image)
cv2.waitKey(0)