import cv2

# Khai báo biến toàn cục
selected_region = []
is_mouse_down = False
cropped_image = None
def draw_rectangle(event, x, y, flags, param):
    global selected_region, is_mouse_down, image

    if event == cv2.EVENT_LBUTTONDOWN:
        is_mouse_down = True
        selected_region = [(x, y)]

    elif event == cv2.EVENT_LBUTTONUP:
        is_mouse_down = False
        selected_region.append((x, y))
        # cv2.rectangle(image, selected_region[0], selected_region[1], (0, 255, 0), 2)
        # cv2.putText(image, f"{selected_region[0]}", selected_region[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Image", image)

        top_left = selected_region[0]
        bottom_right = selected_region[1]

        cropped_image = image[bottom_right[1]:top_left[1], bottom_right[0]:top_left[0]]
        print(bottom_right[1] , top_left[1], bottom_right[0],top_left[0])

        cv2.imshow("name", cropped_image)
        cv2.imwrite("detec_1.png",cropped_image)

        # In ra tọa độ 4 góc
        top_right = (bottom_right[0], top_left[1])
        bottom_left = (top_left[0], bottom_right[1])
        print("Top Left:", top_left)
        print("Top Right:", top_right)
        print("Bottom Right:", bottom_right)
        print("Bottom Left:", bottom_left)

# Đọc ảnh từ đường dẫn
image = cv2.imread("label_1.png")

# Tạo cửa sổ hiển thị ảnh
cv2.namedWindow("Image")
cv2.imshow("Image", image)

# Thiết lập hàm xử lý sự kiện chuột
cv2.setMouseCallback("Image", draw_rectangle)

# Chờ người dùng quét vùng trên ảnh
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Đóng cửa sổ hiển thị
cv2.destroyAllWindows()

