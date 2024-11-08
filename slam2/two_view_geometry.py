import numpy as np
import cv2

def triangulate_points(points1, points2, T1, T2, reprojection_threshold=5e-3):
    #triangulate using linear algebra
    #construct linear system
    points = np.zeros((points1.shape[0], 4))
    A_r = T2[:2, :]
    A_l = T1[:2, :]
    A = np.vstack((A_l, A_r))
    B = np.array([T1[2, :],
        T1[2, :],
        T2[2, :],
        T2[2, :]])
    for i in range(len(points1)):
        x1 = points1[i]
        x2 = points2[i]
        d = np.array([x1[0], x1[1], x2[0], x2[1]])
        S = A - np.diag(d) @ B
        #find nullspace
        _, _, V = np.linalg.svd(S)
        x = V[-1]
        x = x / x[3]
        points[i] = x

    #check reprojection error
    points1_transformed = T1 @ points.T
    points1_reprojected = points1_transformed / points1_transformed[2, :]
    error1 = np.linalg.norm(points1_reprojected[:2, :].T - points1, axis=1)

    points2_transformed = T2 @ points.T
    points2_reprojected = points2_transformed / points2_transformed[2, :]
    error2 = np.linalg.norm(points2_reprojected[:2, :].T - points2, axis=1)

    inliers = np.logical_and(error1 < reprojection_threshold, error2 < reprojection_threshold)

    #check positive depth
    inliers = np.logical_and(inliers, points1_transformed[2, :] > 0, points2_transformed[2, :] > 0)

    #check points far from camera
    inliers = np.logical_and(inliers, points1_transformed[2, :] < 50, points2_transformed[2, :] < 50)
    return points[:, :3], inliers > 0


def two_view_geometry(keypoints1, keypoints2, matches, reprojection_threshold):
    
    points1 = np.array([keypoints1[match[0]] for match in matches])
    points2 = np.array([keypoints2[match[1]] for match in matches])
    keypoints_ids_1 = np.array([match[0] for match in matches])
    keypoints_ids_2 = np.array([match[1] for match in matches])

    E, mask = cv2.findEssentialMat(points1, points2, threshold=reprojection_threshold)
    points1 = points1[mask.flatten() > 0]
    points2 = points2[mask.flatten() > 0]
    keypoints_ids_1 = keypoints_ids_1[mask.flatten() > 0]
    keypoints_ids_2 = keypoints_ids_2[mask.flatten() > 0]


    _, R, t, mask = cv2.recoverPose(E, points1, points2)
    T_c1_c0 = np.hstack((R, t))
    T_c0_c1 = np.hstack((R.T, -R.T @ t))
    points1 = points1[mask.flatten() > 0]
    points2 = points2[mask.flatten() > 0]
    keypoints_ids_1 = keypoints_ids_1[mask.flatten() > 0]
    keypoints_ids_2 = keypoints_ids_2[mask.flatten() > 0]
    T = T_c1_c0

    points, inliers = triangulate_points(points1, points2, np.eye(4), T_c1_c0, reprojection_threshold)

    points = points[inliers]
    keypoints_ids_1 = keypoints_ids_1[inliers]
    keypoints_ids_2 = keypoints_ids_2[inliers]
    return T_c1_c0, points, keypoints_ids_1, keypoints_ids_2


# from ChatGPT

"""
import numpy as np  # Импортируем numpy для работы с массивами и линейной алгеброй
import cv2  # Импортируем OpenCV для вычисления фундаментальной матрицы и позы камеры

# Функция для триангуляции 3D-точек из двух видов (камерных позиций)
def triangulate_points(points1, points2, T1, T2, reprojection_threshold=5e-3):
    # Инициализируем массив для хранения триангулированных точек
    points = np.zeros((points1.shape[0], 4))
    # Извлекаем первые две строки из матриц преобразования T1 и T2
    A_r = T2[:2, :]
    A_l = T1[:2, :]
    A = np.vstack((A_l, A_r))  # Объединяем их в одну матрицу A
    # Создаем матрицу B из третьих строк T1 и T2
    B = np.array([T1[2, :], T1[2, :], T2[2, :], T2[2, :]])
    # Проходим по всем парам соответствующих точек
    for i in range(len(points1)):
        x1 = points1[i]  # Точка на первом изображении
        x2 = points2[i]  # Точка на втором изображении
        d = np.array([x1[0], x1[1], x2[0], x2[1]])
        S = A - np.diag(d) @ B  # Строим систему уравнений для каждой точки
        # Находим решение с использованием SVD, чтобы получить гомогенную координату
        _, _, V = np.linalg.svd(S)
        x = V[-1]
        x = x / x[3]  # Нормализуем координаты
        points[i] = x

    # Проверяем ошибку проекции
    points1_transformed = T1 @ points.T
    points1_reprojected = points1_transformed / points1_transformed[2, :]
    error1 = np.linalg.norm(points1_reprojected[:2, :].T - points1, axis=1)

    points2_transformed = T2 @ points.T
    points2_reprojected = points2_transformed / points2_transformed[2, :]
    error2 = np.linalg.norm(points2_reprojected[:2, :].T - points2, axis=1)

    # Определяем инлайеры на основе пороговой ошибки
    inliers = np.logical_and(error1 < reprojection_threshold, error2 < reprojection_threshold)

    # Проверяем, находятся ли точки перед камерами (положительная глубина)
    inliers = np.logical_and(inliers, points1_transformed[2, :] > 0, points2_transformed[2, :] > 0)

    # Проверяем, находятся ли точки в пределах заданного расстояния
    inliers = np.logical_and(inliers, points1_transformed[2, :] < 50, points2_transformed[2, :] < 50)
    # Возвращаем триангулированные 3D-точки и инлайеры
    return points[:, :3], inliers > 0

# Функция для вычисления геометрии между двумя изображениями
def two_view_geometry(keypoints1, keypoints2, matches, reprojection_threshold):
    # Извлекаем координаты соответствующих точек по матчам
    points1 = np.array([keypoints1[match[0]] for match in matches])
    points2 = np.array([keypoints2[match[1]] for match in matches])
    # Извлекаем индексы ключевых точек
    keypoints_ids_1 = np.array([match[0] for match in matches])
    keypoints_ids_2 = np.array([match[1] for match in matches])

    # Находим матрицу E (Essential Matrix) с маской инлайеров
    E, mask = cv2.findEssentialMat(points1, points2, threshold=reprojection_threshold)
    # Фильтруем точки по инлайерам
    points1 = points1[mask.flatten() > 0]
    points2 = points2[mask.flatten() > 0]
    keypoints_ids_1 = keypoints_ids_1[mask.flatten() > 0]
    keypoints_ids_2 = keypoints_ids_2[mask.flatten() > 0]

    # Восстанавливаем ротацию R и трансляцию t между камерами
    _, R, t, mask = cv2.recoverPose(E, points1, points2)
    # Создаем матрицы преобразования из камеры c0 в c1 и обратно
    T_c1_c0 = np.hstack((R, t))
    T_c0_c1 = np.hstack((R.T, -R.T @ t))
    # Фильтруем только инлайнеры после восстановления позы
    points1 = points1[mask.flatten() > 0]
    points2 = points2[mask.flatten() > 0]
    keypoints_ids_1 = keypoints_ids_1[mask.flatten() > 0]
    keypoints_ids_2 = keypoints_ids_2[mask.flatten() > 0]
    T = T_c1_c0

    # Триангулируем 3D-точки, используя геометрию двух видов
    points, inliers = triangulate_points(points1, points2, np.eye(4), T_c1_c0, reprojection_threshold)

    # Оставляем только инлайнеры после триангуляции
    points = points[inliers]
    keypoints_ids_1 = keypoints_ids_1[inliers]
    keypoints_ids_2 = keypoints_ids_2[inliers]
    # Возвращаем матрицу преобразования и триангулированные точки
    return T_c1_c0, points, keypoints_ids_1, keypoints_ids_2





"""