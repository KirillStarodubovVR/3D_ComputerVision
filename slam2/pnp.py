import numpy as np
import cv2

def solve_pnp(points, landmark_ids, keypoints, keypoints_ids, reprojection_threshold=5e-3):
    #solve pnp problem using 3d points and 2d keypoints

    #get 3d points and 2d keypoints
    points = points[landmark_ids]
    keypoints = np.array([keypoints[keypoints_id] for keypoints_id in keypoints_ids])

    #solve pnp
    _, rvec, tvec, inliers = cv2.solvePnPRansac(points, keypoints, np.eye(3), np.zeros(5), confidence=0.99, reprojectionError=reprojection_threshold, iterationsCount=1000)
    if inliers is None or len(inliers) < 100:
        print('No inliers found')
        return None, None, None
    print('Found {} inliers'.format(len(inliers)))
    R, _ = cv2.Rodrigues(rvec)
    T = np.hstack((R, tvec))

    #filter outliers
    landmark_ids = np.array(landmark_ids)[inliers.flatten()]
    keypoints_ids = np.array(keypoints_ids)[inliers.flatten()]

    return T, landmark_ids, keypoints_ids


# from ChatGPT

"""
import tqdm  # Импортируем tqdm для отображения прогресса
import os  # Импортируем os для работы с файловой системой
import random  # Импортируем random для случайного перемешивания списка
from tqdm import tqdm  # Импортируем tqdm для отображения прогресса загрузки
import cv2  # Импортируем OpenCV для работы с изображениями

# Функция для получения списка путей к файлам изображений в указанной директории
def get_image_files(dir, max_views=-1, shuffle=False):
    # Находим все файлы с расширением '.JPG' в директории и создаем список их путей
    files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.JPG')]
    # Если установлен флаг shuffle, перемешиваем список случайным образом
    if shuffle:
        random.shuffle(files)
    # Если задано максимальное количество файлов для загрузки, ограничиваем список
    if max_views > 0:
        files = files[:max_views]
    # Возвращаем список файлов
    return files

# Функция для загрузки изображений из списка файлов
def load_images(files, resize_factor=1.0):
    # Инициализируем пустой список для изображений
    images = []
    # Загружаем каждое изображение и отображаем прогресс
    for file in tqdm(files, desc='Loading images'):
        # Читаем изображение из файла и добавляем его в список
        images.append(cv2.imread(file))
    # Если коэффициент изменения размера не равен 1.0, изменяем размер изображений
    if resize_factor != 1.0:
        images = [
            cv2.resize(
                image, 
                (int(image.shape[1] * resize_factor), int(image.shape[0] * resize_factor))
            ) 
            for image in images
        ]
    # Возвращаем список изображений
    return images
"""