import cv2
from PIL import Image, ImageDraw, ImageFont

def opencv():
    img_label1 = cv2.imread("label_1.png")

    positions = [
        (43, 48, 54, 216),
        (31, 62, 41, 195),
        (22, 48, 32, 208),
        (11, 64, 23, 193),
        (23, 227, 35, 296),
        (35, 226, 46, 300),
        (5, 4, 17, 37)
    ]

    for i, pos in enumerate(positions, start=1):
        y1, x1, y2, x2 = pos
        cv2.putText(img_label1, str(i), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Labeled Image", img_label1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def PIL():
    img_label1 = Image.open("label_1.png")

    draw = ImageDraw.Draw(img_label1)

    positions = [
        (43, 48, 54, 216),
        (31, 62, 41, 195),
        (22, 48, 32, 208),
        (11, 64, 23, 193),
        (23, 227, 35, 296),
        (35, 226, 46, 300),
        (5, 4, 17, 37)
    ]
    font = ImageFont.truetype("arial.ttf", 12)
    fill = (0, 0, 255)
    thickness = 2

    for i, pos in enumerate(positions, start=1):
        y1, x1, y2, x2 = pos
        draw.text((x1, y1), str(i), font=font, fill=fill, stroke_width=thickness)

    img_label1.show()

PIL()
