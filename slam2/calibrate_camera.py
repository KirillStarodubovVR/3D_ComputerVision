import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# path_to_images = os.path.join(os.getcwd(), "images_calibration")
path_to_images = "./images/calibrate"
# Критерий остонова
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Подготовка точек объекта в виде (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((8 * 6, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

# Инициализируем массивы для хранения точек объектов и изображения.
objpoints = []  # трехмерная точка в реальном пространстве
imgpoints = []  # двумерные точки на плоскости изображения.

images = os.listdir(path_to_images)
# Сортировка списка, чтобы исключить откалиброванные изображения
images = [image for image in images if "calib_" not in image]
print(f"По указанному пути найденно {len(images)} изображений")

for i, fname in enumerate(images):
    img = cv2.imread(os.path.join(path_to_images, fname))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Ищем углы шахматной доски
    ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)

    # Если найдено, добавьте точки объекта, точки изображения (после их уточнения).
    if ret:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("Матрица камеры:")
print(mtx)
print("Коэффициенты искажений:")
print(dist)


img = cv2.imread(os.path.join(path_to_images, images[6]))
h, w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
print("Новая матрица камеры:")
print(newcameramtx)
print("ROI:")
print(roi)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print("total error: ", mean_error/len(objpoints))