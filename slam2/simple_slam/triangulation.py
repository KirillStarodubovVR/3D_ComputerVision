import cv2
import numpy as np
"""

"""
def triangulate_new_points(frame1, frame2, K, Tw_f1, Tw_f2, feature_coverage = None):

    # Создаем объект SIFT для детекции и извлечения признаков
    sift = cv2.SIFT_create()
    # Находим ключевые точки и дескрипторы для первого изображения
    kp1, des1 = sift.detectAndCompute(frame1, None)
    # Находим ключевые точки и дескрипторы для второго изображения
    kp2, des2 = sift.detectAndCompute(frame2, None)


    # Проверка и фильтрация признаков на втором изображении, если feature_coverage задан
    if feature_coverage is not None:
        # Определяем размерности сетки для покрытия фич
        h, w = feature_coverage.shape
        # Рассчитываем соотношения (размеры ячеек) по осям Y и X
        rY = frame1.shape[0] // h
        rX = frame1.shape[1] // w
        # Оставляем только те фичи, которые находятся в ячейках, где feature_coverage == 0
        filtered_features = [i for i, kp in enumerate(kp2)
                             if feature_coverage[int(kp.pt[1]) // rY,
                                                 int(kp.pt[0]) // rX] == 0]
        # Обновляем ключевые точки и дескрипторы для второго изображения после фильтрации
        kp2 = [kp2[i] for i in filtered_features]
        des2 = des2[filtered_features, :]

    flann = cv2.FlannBasedMatcher(dict(algorithm=0, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = [m for m in matches if m[0].distance < 0.7 * m[1].distance]

    if not good_matches:
        return np.empty((0, 3)), np.empty((0, 128))

    src_pts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    P1 = K @ Tw_f1
    P2 = K @ Tw_f2
    points_4D = cv2.triangulatePoints(P1, P2, src_pts.reshape(-1, 2).T, dst_pts.reshape(-1, 2).T)
    points_3D = points_4D / points_4D[3]

    # проверяем, что точки триангулировались корректно
    # (т.е не соответвуют "мусорным" матчам)
    depths2 = (Tw_f2[:, :3] @ points_3D[:3, :] + Tw_f2[:, 3].reshape(3, 1))[2]

    #глубина должны быть положительная
    mask = depths2 > 0

    projected_pts1 = (P1 @ points_4D)[:2] / (P1 @ points_4D)[2]
    projected_pts2 = (P2 @ points_4D)[:2] / (P2 @ points_4D)[2]

    reprojection_error1 = np.sum((src_pts.reshape(-1, 2) - projected_pts1.T)**2, axis=1)
    reprojection_error2 = np.sum((dst_pts.reshape(-1, 2) - projected_pts2.T)**2, axis=1)
    error_threshold = 5

    # точки должны репроецироваться в задетекченные фичи
    mask = (mask & (reprojection_error1 < error_threshold) &
            (reprojection_error2 < error_threshold))

    print ("Added", mask.sum(), "points", sep=" ")

    descriptors = np.array([des2[m[0].trainIdx] for m in good_matches])

    points_3D = points_3D[:, mask]
    descriptors = descriptors[mask, :]
    src_pts = src_pts[mask, :, :]
    dst_pts = dst_pts[mask, :, :]

    return points_3D[:3].T, descriptors, src_pts, dst_pts

"""
def triangulate_new_points(frame1, frame2, K, Tw_f1, Tw_f2, feature_coverage=None):
    # Создаем объект SIFT для детекции и извлечения признаков
    sift = cv2.SIFT_create()
    # Находим ключевые точки и дескрипторы для первого изображения
    kp1, des1 = sift.detectAndCompute(frame1, None)
    # Находим ключевые точки и дескрипторы для второго изображения
    kp2, des2 = sift.detectAndCompute(frame2, None)

    # Проверка и фильтрация признаков на втором изображении, если feature_coverage задан
    if feature_coverage is not None:
        # Определяем размерности сетки для покрытия фич
        h, w = feature_coverage.shape
        # Рассчитываем соотношения (размеры ячеек) по осям Y и X
        rY = frame1.shape[0] // h
        rX = frame1.shape[1] // w
        # Оставляем только те фичи, которые находятся в ячейках, где feature_coverage == 0
        filtered_features = [i for i, kp in enumerate(kp2)
                             if feature_coverage[int(kp.pt[1]) // rY,
                                                 int(kp.pt[0]) // rX] == 0]
        # Обновляем ключевые точки и дескрипторы для второго изображения после фильтрации
        kp2 = [kp2[i] for i in filtered_features]
        des2 = des2[filtered_features, :]

    # Создаем FLANN-матчер для поиска соответствий между дескрипторами
    flann = cv2.FlannBasedMatcher(dict(algorithm=0, trees=5), dict(checks=50))
    # Находим k=2 ближайших соседей для каждой пары дескрипторов
    matches = flann.knnMatch(des1, des2, k=2)
    # Применяем Lowe's ratio test для фильтрации хороших совпадений
    good_matches = [m for m in matches if m[0].distance < 0.7 * m[1].distance]

    # Проверка: если нет хороших совпадений, возвращаем пустые массивы
    if not good_matches:
        return np.empty((0, 3)), np.empty((0, 128))

    # Извлекаем координаты точек в пикселях для каждой пары совпадений
    src_pts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Формируем матрицы проекций для двух кадров
    P1 = K @ Tw_f1
    P2 = K @ Tw_f2
    # Выполняем триангуляцию для получения 4D-гомогенных координат
    points_4D = cv2.triangulatePoints(P1, P2, src_pts.reshape(-1, 2).T, dst_pts.reshape(-1, 2).T)
    # Переходим от 4D-гомогенных координат к 3D, нормируя по последней координате
    points_3D = points_4D / points_4D[3]

    # Рассчитываем глубины для второго кадра, чтобы проверить положение точек перед камерой
    depths2 = (Tw_f2[:, :3] @ points_3D[:3, :] + Tw_f2[:, 3].reshape(3, 1))[2]

    # Фильтр: точки должны быть перед камерой (глубина положительна)
    mask = depths2 > 0

    # Проецируем 4D-точки обратно на изображение для проверки репроекционных ошибок
    projected_pts1 = (P1 @ points_4D)[:2] / (P1 @ points_4D)[2]
    projected_pts2 = (P2 @ points_4D)[:2] / (P2 @ points_4D)[2]

    # Считаем ошибки репроекции для каждой точки в обоих кадрах
    reprojection_error1 = np.sum((src_pts.reshape(-1, 2) - projected_pts1.T)**2, axis=1)
    reprojection_error2 = np.sum((dst_pts.reshape(-1, 2) - projected_pts2.T)**2, axis=1)
    error_threshold = 5  # Порог для допустимой репроекционной ошибки

    # Фильтр: оставляем точки, у которых репроекционная ошибка меньше порога
    mask = (mask & (reprojection_error1 < error_threshold) &
            (reprojection_error2 < error_threshold))

    # Выводим количество добавленных точек
    print("Added", mask.sum(), "points", sep=" ")

    # Извлекаем дескрипторы из второго изображения для хороших совпадений
    descriptors = np.array([des2[m[0].trainIdx] for m in good_matches])

    # Применяем маску для фильтрации 3D-точек, дескрипторов и координат
    points_3D = points_3D[:, mask]
    descriptors = descriptors[mask, :]
    src_pts = src_pts[mask, :, :]
    dst_pts = dst_pts[mask, :, :]

    # Возвращаем 3D-точки, дескрипторы и координаты совпавших точек
    return points_3D[:3].T, descriptors, src_pts, dst_pts
"""