import tqdm
import os
import random
from tqdm import tqdm
import cv2
import re # new

# Функция для получения списка путей к файлам изображений в указанной директории
def get_image_files(dir, max_views=-1, shuffle=False, sort=True):
    # Находим все файлы с расширением '.JPG' в директории и создаем список их путей
    files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.JPG')]
    if sort:
        # Упорядочивание кадров для исследования корреляций между кадрами
        # files = sorted(files, key=lambda x: int(re.search(r'frame_(\d+)', x).group(1))) # delete
        files.sort()

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
        image = cv2.imread(file)
        if resize_factor != 1.0:
            image = cv2.resize(image,
                               (int(image.shape[1] * resize_factor),
                                int(image.shape[0] * resize_factor))
                               )

        images.append(image)


    # Если коэффициент изменения размера не равен 1.0, изменяем размер изображений
    # if resize_factor != 1.0:
    #     images = [cv2.resize(image,
    #                          (int(image.shape[1] * resize_factor),
    #                           int(image.shape[0] * resize_factor))
    #                          )
    #               for image in images]

    # Возвращаем список изображений
    return images
