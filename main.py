import cv2
import os

cap = cv2.VideoCapture(0)

num = 0

while cap.isOpened():

    succes, img = cap.read()

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit

        image_path = os.path.join(os.getcwd(), 'images_calibration', f"{num} + '.png'")
        cv2.imwrite(image_path, img)
        print("image saved!")
        num += 1

    cv2.imshow('Img 1',img)