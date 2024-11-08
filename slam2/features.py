from tqdm import tqdm
import cv2
import numpy as np

def extract_features(images, num_features):
    features = []
    for image in tqdm(images, desc='Extracting features'):
        #detect keypoints
        detector = cv2.SIFT_create(nfeatures = num_features)
        keypoints, descriptors = detector.detectAndCompute(image, None)
        #transform keypoints to numpy array
        keypoints = np.array([keypoint.pt for keypoint in keypoints])
        features.append((keypoints, descriptors))
    return np.array(features, dtype=object)

def match_features(features):
    # match features using flann, match each image to every other image
    matches = []
    for i in tqdm(range(len(features)), desc='Matching features'):
        matches.append([])
        for j in range(len(features)):
            if i == j:
                matches[i].append(None)
                continue
            #create flann matcher
            matcher = cv2.FlannBasedMatcher_create()
            #find matches
            m = matcher.knnMatch(features[i][1], features[j][1], k=2)
            good = []
            for m1, m2 in m:
                if m1.distance < 0.7 * m2.distance:
                    #convert match to tupple
                    good.append((m1.queryIdx, m1.trainIdx))
            matches[i].append(good)

    return np.array(matches, dtype=object)

def cross_check(matches):
    # cross-check matches, only keep matches that are mutual

    for i in range(len(matches)):
        for j in range(len(matches[i])):
            if matches[i][j] is None:
                continue
            matches[i][j] = [match for match in matches[i][j] if match in [(match[1], match[0]) for match in matches[j][i]]]
    return matches


# from ChatGPT
"""
from tqdm import tqdm  # Импортируем библиотеку tqdm для отображения прогресса
import cv2  # Импортируем библиотеку OpenCV для обработки изображений и извлечения признаков
import numpy as np  # Импортируем библиотеку numpy для работы с массивами

# Функция для извлечения признаков из списка изображений
def extract_features(images, num_features):
    # Инициализируем список для хранения признаков
    features = []
    # Проходим по каждому изображению, отображая прогресс
    for image in tqdm(images, desc='Extracting features'):
        # Создаем детектор признаков SIFT с заданным количеством признаков
        detector = cv2.SIFT_create(nfeatures=num_features)
        # Извлекаем ключевые точки и дескрипторы
        keypoints, descriptors = detector.detectAndCompute(image, None)
        # Преобразуем ключевые точки в массив numpy
        keypoints = np.array([keypoint.pt for keypoint in keypoints])
        # Добавляем пары ключевых точек и дескрипторов в список features
        features.append((keypoints, descriptors))
    # Возвращаем массив признаков
    return np.array(features, dtype=object)

# Функция для сопоставления признаков между всеми изображениями
def match_features(features):
    # Инициализируем список для хранения совпадений
    matches = []
    # Проходим по всем комбинациям изображений, отображая прогресс
    for i in tqdm(range(len(features)), desc='Matching features'):
        # Создаем пустой список для совпадений с текущим изображением
        matches.append([])
        # Сравниваем текущее изображение со всеми другими изображениями
        for j in range(len(features)):
            # Если изображение сравнивается с самим собой, добавляем None
            if i == j:
                matches[i].append(None)
                continue
            # Создаем объект для сопоставления признаков методом FLANN
            matcher = cv2.FlannBasedMatcher_create()
            # Находим совпадения с помощью метода k-ближайших соседей (k=2)
            m = matcher.knnMatch(features[i][1], features[j][1], k=2)
            # Список для хороших совпадений
            good = []
            # Фильтруем совпадения с использованием условия Lowe's ratio test
            for m1, m2 in m:
                if m1.distance < 0.7 * m2.distance:
                    # Добавляем хорошие совпадения в список (индексы совпадений)
                    good.append((m1.queryIdx, m1.trainIdx))
            # Добавляем совпадения для текущей пары изображений
            matches[i].append(good)
    # Возвращаем массив совпадений
    return np.array(matches, dtype=object)

# Функция для выполнения перекрестной проверки совпадений
def cross_check(matches):
    # Проходим по каждой паре изображений
    for i in range(len(matches)):
        for j in range(len(matches[i])):
            # Пропускаем, если совпадений нет
            if matches[i][j] is None:
                continue
            # Оставляем только взаимные совпадения (которые встречаются в обоих направлениях)
            matches[i][j] = [
                match for match in matches[i][j]
                if match in [(m[1], m[0]) for m in matches[j][i]]
            ]
    # Возвращаем отфильтрованные совпадения
    return matches
"""