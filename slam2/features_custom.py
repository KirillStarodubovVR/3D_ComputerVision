"""
Предложенные режимы сопоставления фичей в COLORMAP, предназначенных для различных сценариев ввода данных:

Exhaustive Matching: Если количество изображений в вашем наборе данных относительно невелико (до нескольких сотен), 
этот режим должен быть достаточно быстрым и приводить к наилучшим результатам реконструкции. Здесь каждое изображение 
сопоставляется с каждым другим изображением, а размер блока определяет, сколько изображений одновременно загружается с диска в память.

Sequential Matching: этот режим полезен, если изображения получены в последовательном порядке, например, с помощью видеокамеры. 
В этом случае последовательные кадры визуально перекрываются, и нет необходимости в исчерпывающем сопоставлении всех пар изображений. 
Вместо этого последовательно снятые изображения сопоставляются друг с другом. Этот режим сопоставления имеет встроенную функцию обнаружения петель 
на основе словарного дерева, где каждое N-е изображение (loop_detection_period) сопоставляется с наиболее визуально
похожих изображений (loop_detection_num_images). Обратите внимание, что имена файлов изображений должны быть упорядочены
последовательно (например, image0001.jpg, image0002.jpg и т. д.).
Порядок в базе данных не имеет значения, так как изображения явно упорядочены в соответствии с именами файлов. 
Обратите внимание, что для обнаружения петель требуется предварительно обученное дерево лексики, которое можно загрузить с сайта https://demuc.de/colmap/.

Vocabulary Tree Matching: в этом режиме [schoenberger16vote] каждое изображение сопоставляется со своими ближайшими визуальными
соседей, используя дерево словарей с пространственным ранжированием. Это рекомендуемый режим сопоставления для больших коллекций изображений
коллекций (несколько тысяч). Для этого требуется предварительно обученное дерево лексики, которое можно загрузить с сайта https://demuc.de/colmap/.

Spatial Matching: в этом режиме каждое изображение сопоставляется со своими пространственными ближайшими соседями. Пространственное расположение
могут быть заданы вручную в управлении базой данных. По умолчанию COLMAP также извлекает GPS-информацию из EXIF и использует ее
ее для поиска ближайших пространственных соседей. Если имеется точная предварительная информация о местоположении, рекомендуется использовать именно этот режим сопоставления.
Транзитивное сопоставление: этот режим сопоставления использует транзитивные отношения уже существующих совпадений признаков для
чтобы создать более полный граф соответствия. Если изображение A совпадает с изображением B, а B - C, то этот механизм сопоставления
пытается сопоставить A с C напрямую.

Custom Matching: этот режим позволяет задавать отдельные пары изображений для сопоставления или импортировать отдельные признаки
совпадения. Чтобы указать пары изображений, необходимо предоставить текстовый файл с одной парой изображений в строке.
"""

# Для данной работы возьмём Exhaustive Matching по причине эффективности для малого количества изображений.


import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


# def extract_features(image, num_features):
#     """
#     Извлекает ключевые точки и дескрипторы из изображения.
#     """
#     detector = cv2.SIFT_create(nfeatures = num_features)
#     keypoints, descriptors = detector.detectAndCompute(image, None)
#     return keypoints, descriptors

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

def match_features(descriptor1, descriptor2, ratio_thresh=0.8):
    """
    Выполняет сопоставление дескрипторов методом KNN с фильтрацией по порогу Лоу.
    """
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn_matches = matcher.knnMatch(descriptor1, descriptor2, k=2)
    good = [(m.queryIdx, m.trainIdx) for m, n in knn_matches if m.distance < ratio_thresh * n.distance]
    return good




def exhaustive_nearest_neighbor_matching(features, num_neighbors=5):
    """
    Выполняет сопоставление изображений с ограничением по ближайшим соседям,
    используя подход Exhaustive Matching и Nearest Neighbor.
    """
    # Извлекаем дескрипторы для всех изображений
    descriptors_list = features[:, 1]
    keypoints_list = features[:, 0]

    # Выполняем Nearest Neighbor для поиска ближайших соседей среди дескрипторов
    all_descriptors = np.vstack(descriptors_list)
    nn = NearestNeighbors(n_neighbors=num_neighbors, algorithm='auto').fit(all_descriptors)
    print("Ищем ближайших соседей")
    _, indices = nn.kneighbors(all_descriptors)

    dict_descriptors = {}
    last_index = 0
    for i, d in enumerate(descriptors_list):
        l = len(d)
        for j in range(last_index, l+last_index):
            dict_descriptors[j] = i
        last_index = l+last_index

    matches = [[[] for _ in range(len(descriptors_list))] for _ in range(len(descriptors_list))]

    for i, des1 in tqdm(enumerate(all_descriptors)):
        i1 = dict_descriptors.get(i)
        # Создаем пустой список для совпадений с текущим изображением
        for j in indices[i]:  # Проходим по ближайшим соседям
            i2 = dict_descriptors.get(j)
            if i1 >= i2:  # Избегаем повторов и самосопоставлений
                continue

            des2 = all_descriptors[j]
            good = match_features(des1, des2)
            if good:
                matches[i1][i2].append(good)

                # img_matches.append((i, j, good_matches))
        # matches.append(img_matches)

    return np.array(matches, dtype=object)