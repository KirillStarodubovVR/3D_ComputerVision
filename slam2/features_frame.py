from tqdm import tqdm
import cv2
import numpy as np

def extract_features(images, num_features):
    # Функция для извлечения признаков из списка изображений
    # Инициализируем список для хранения признаков
    features = []
    # Проходим по каждому изображению, отображая прогресс
    for image in tqdm(images, desc='Extracting features'):
        # Создаем детектор признаков SIFT с заданным количеством признаков
        detector = cv2.SIFT_create(nfeatures = num_features)
        # Извлекаем ключевые точки и дескрипторы
        keypoints, descriptors = detector.detectAndCompute(image, None)
        # Преобразуем ключевые точки в массив numpy
        keypoints = np.array([keypoint.pt for keypoint in keypoints])
        # Добавляем пары ключевых точек и дескрипторов в список features
        features.append((keypoints, descriptors))

    # Возвращаем массив признаков
    return np.array(features, dtype=object)

def match_features(features, frame=10):
    # Функция для сопоставления признаков между всеми изображениями
    # Инициализируем список для хранения совпадений
    # matches = [[[] for _ in range(len(features))] for _ in range(len(features))]
    matches = []
    num_images = len(features)
    # Проходим по всем комбинациям изображений, отображая прогресс
    for i in tqdm(range(num_images), desc='Matching features'):
        # Создаем пустой список для совпадений с текущим изображением
        matches.append([])

        # Задаём границы окна
        start = i - frame
        end = i + frame + 1

        # Сравниваем текущее изображение со всеми другими изображениями
        for j in range(num_images):
            frame_indecies = [i%num_images for i in range(start,end)]
            # Если изображение сравнивается с самим собой, добавляем None
            if i==j or (j not in frame_indecies):
                matches[i].append(None)
                continue
            # Создаем объект для сопоставления признаков методом FLANN
            matcher = cv2.FlannBasedMatcher_create()
            # Находим совпадения с помощью метода k-ближайших соседей (k=2)
            m = matcher.knnMatch(features[i][1], features[j%num_images][1], k=2)
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

def cross_check(matches):
    # Функция для выполнения перекрестной проверки совпадени
    # Проходим по каждой паре изображений
    for i in tqdm(range(len(matches)), "Cross check:"):
        for j in range(len(matches[i])):
            # Пропускаем, если совпадений нет (одно и тоже изображение)
            if matches[i][j] is None:
                continue
            # Оставляем только взаимные совпадения (которые встречаются в обоих направлениях)
            matches[i][j] = [match for match in matches[i][j] if match in [(match[1], match[0]) for match in matches[j][i]]]
    # Возвращаем отфильтрованные совпадения
    return matches
