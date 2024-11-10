import numpy as np
import cv2

def triangulate_points(points1, points2, T1, T2, reprojection_threshold=5e-3):
    # Триангулирует 3D-точки на основе соответствующих точек из двух видов с помощью линейной алгебры
    # Инициализируем массив для хранения 3D-точек в виде однородных координат
    points = np.zeros((points1.shape[0], 4))    # Каждая точка имеет 4 координаты (x, y, z, w)
    # Подготовка матриц для решения системы уравнений
    A_r = T2[:2, :] # Первая и вторая строки матрицы преобразования T2 (для второго вида)
    A_l = T1[:2, :] # Первая и вторая строки матрицы преобразования T1 (для первого вида)
    A = np.vstack((A_l, A_r))   # Объединяем строки A_l и A_r в одну матрицу A
    # Создаем матрицу B из третьей строки T1 и T2, которая используется для построения системы уравнений
    B = np.array([T1[2, :], T1[2, :], T2[2, :], T2[2, :]])
    # Проходим по каждой паре соответствующих точек
    for i in range(len(points1)):
        x1 = points1[i] # Координаты точки в первом виде
        x2 = points2[i] # Координаты точки во втором виде
        d = np.array([x1[0], x1[1], x2[0], x2[1]]) # Объединяем координаты x и y из обеих точек
        # Формируем матрицу S для текущей пары точек
        S = A - np.diag(d) @ B
        # Находим решение уравнения с помощью SVD, где x — 3D-точка в однородных координатах
        _, _, V = np.linalg.svd(S)
        x = V[-1]  # Берем последнюю строку V (решение)
        x = x / x[3]  # Нормализуем точку, чтобы ее четвертая координата была равна 1
        points[i] = x  # Сохраняем точку

    # Проверяем ошибки репроекции
    points1_transformed = T1 @ points.T # Преобразуем точки в систему первой камеры
    points1_reprojected = points1_transformed / points1_transformed[2, :]   # Нормализуем по z-координате
    error1 = np.linalg.norm(points1_reprojected[:2, :].T - points1, axis=1) # Ошибка репроекции в первом виде

    points2_transformed = T2 @ points.T # Преобразуем точки в систему второй камеры
    points2_reprojected = points2_transformed / points2_transformed[2, :]   # Нормализуем по z-координате
    error2 = np.linalg.norm(points2_reprojected[:2, :].T - points2, axis=1) # Ошибка репроекции во втором виде
    # Определяем инлиеры: точки с ошибкой репроекции, не превышающей заданный порог
    inliers = np.logical_and(error1 < reprojection_threshold, error2 < reprojection_threshold)

    # Проверяем, чтобы точки находились перед камерами (глубина положительна)
    inliers = np.logical_and(inliers, points1_transformed[2, :] > 0, points2_transformed[2, :] > 0)

    # Проверяем, чтобы точки находились в разумном расстоянии от камеры
    inliers = np.logical_and(inliers, points1_transformed[2, :] < 50, points2_transformed[2, :] < 50)
    # Возвращаем 3D-координаты точек (x, y, z) и массив инлиеров
    return points[:, :3], inliers > 0


def two_view_geometry(keypoints1, keypoints2, matches, reprojection_threshold):
    """
    Подготовка соответствующих точек для вычисления матрицы Essential,
    до применения триангуляции для получения 3D-точек.
    """
    # Функция вычисляет геометрию между двумя видами на основе соответствий ключевых точек.
    # Преобразуем соответствия в массивы координат ключевых точек для каждого изображения.
    points1 = np.array([keypoints1[match[0]] for match in matches]) # Координаты точек в первом изображении
    points2 = np.array([keypoints2[match[1]] for match in matches]) # Координаты точек во втором изображении

    # Сохраняем индексы ключевых точек для каждого изображения
    keypoints_ids_1 = np.array([match[0] for match in matches]) # Индексы ключевых точек в первом изображении
    keypoints_ids_2 = np.array([match[1] for match in matches]) # Индексы ключевых точек во втором изображении

    # Находим матрицу E (основная матрица) для двух изображений. Метод cv2.RANSAC
    E, mask = cv2.findEssentialMat(points1, points2, threshold=reprojection_threshold)
    # Фильтруем точки на основании маски, полученной из cv2.findEssentialMat
    points1 = points1[mask.flatten() > 0]   # Точки в первом изображении, прошедшие фильтр
    points2 = points2[mask.flatten() > 0]   # Точки во втором изображении, прошедшие фильтр
    keypoints_ids_1 = keypoints_ids_1[mask.flatten() > 0]  # Индексы ключевых точек первого изображения, прошедшие фильтр
    keypoints_ids_2 = keypoints_ids_2[mask.flatten() > 0]  # Индексы ключевых точек второго изображения, прошедшие фильтр

    # Извлекаем относительное вращение и трансляцию между двумя изображениями
    # По факту мы знаем матрицу K но сейчас она нам не нужна
    _, R, t, mask = cv2.recoverPose(E, points1, points2)
    # Формируем матрицу преобразования от камеры 1 к камере 0
    T_c1_c0 = np.hstack((R, t))  # Матрица преобразования от камеры 0 к камере 1.
    T_c0_c1 = np.hstack((R.T, -R.T @ t)) # Формируем матрицу обратного преобразования от камеры 1 к камере 2.
    # Фильтруем точки на основе новой маски, полученной из recoverPose
    # Оставляем только точки, прошедшие фильтрацию с новой маской
    points1 = points1[mask.flatten() > 0]
    points2 = points2[mask.flatten() > 0]
    keypoints_ids_1 = keypoints_ids_1[mask.flatten() > 0]
    keypoints_ids_2 = keypoints_ids_2[mask.flatten() > 0]

    # Выбираем матрицу преобразования от камеры 0 к камере 1 для триангуляции
    T = T_c1_c0 # матрица перехода от камеры 0 к камере 1
    # Триангулируем 3D-точки на основе соответствующих точек из двух изображений
    points, inliers = triangulate_points(points1, points2, np.eye(4), T_c1_c0, reprojection_threshold)

    # Фильтруем только те 3D-точки, которые являются инлаиерами
    points = points[inliers] # Оставляем только инлаиеры
    keypoints_ids_1 = keypoints_ids_1[inliers]
    keypoints_ids_2 = keypoints_ids_2[inliers]
    # Возвращаем матрицу преобразования, 3D-точки и индексы ключевых точек, которые были сопоставлены
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