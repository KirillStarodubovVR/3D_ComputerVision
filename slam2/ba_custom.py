import numpy as np
from scipy.optimize import least_squares
import cv2

# Функция для проецирования 3D-точки на 2D
def project_point(point_3D, pose, K):
    """
    Проецирует 3D-точку на 2D с помощью текущей позы камеры и матрицы калибровки.
    """
    # Извлекаем поворот R и трансляцию t из позы камеры
    R, t = pose[:3, :3], pose[:3, 3]
    # Преобразуем точку в координаты камеры
    point_cam = R @ point_3D + t
    # Проецируем в 2D-координаты с нормализацией
    point_proj = K @ (point_cam / point_cam[2])
    # Возвращаем координаты точки в пикселях
    return point_proj[:2]

# Функция для вычисления ошибки репроекции
def reprojection_residual(params, num_frames, num_points, K, observations):
    """
    Вычисляет ошибку репроекции для всех точек и всех кадров.
    """
    # Извлекаем позы из вектора параметров
    poses = params[:num_frames * 6].reshape((num_frames, 6))
    # Извлекаем 3D-точки из вектора параметров
    points_3D = params[num_frames * 6:].reshape((num_points, 3))

    # Список для накопления ошибок репроекции
    residuals = []
    # Перебираем наблюдения для каждого кадра и каждой точки
    for (frame_id, point_id), observation in observations.items():
        # Извлекаем текущую позу кадра
        pose = poses[frame_id]
        rvec, tvec = pose[:3], pose[3:]

        # Преобразуем в матрицу поворота R
        R, _ = cv2.Rodrigues(rvec)
        # Объединяем поворот и трансляцию в матрицу
        pose_mat = np.hstack((R, tvec.reshape(3, 1)))

        # Проецируем 3D-точку на 2D
        point_proj = project_point(points_3D[point_id], pose_mat, K)

        # Вычисляем ошибку как разность между проекцией и наблюдением
        error = point_proj - observation
        # Добавляем ошибки по x и y в список
        residuals.extend(error)

    return np.array(residuals)  # Возвращаем массив всех ошибок

# Функция для выполнения Bundle Adjustment
def bundle_adjustment(poses, map_points, K, observations):
    """
    Выполняет Bundle Adjustment с помощью scipy.optimize.least_squares.
    """
    num_frames = len(poses)  # Число кадров
    num_points = len(map_points)  # Число 3D-точек

    # Формируем начальный вектор параметров из поз и 3D-точек
    initial_params = []
    for pose in poses.values():
        # Конвертируем поворот из матрицы в вектор с помощью Rodrigues
        rvec, _ = cv2.Rodrigues(pose[:3, :3])
        # Извлекаем вектор трансляции
        tvec = pose[:3, 3]
        # Добавляем к начальным параметрам
        initial_params.extend(rvec.squeeze())
        initial_params.extend(tvec.squeeze())

    # Добавляем координаты всех 3D-точек к начальным параметрам
    initial_params.extend(map_points.flatten())

    # Запускаем оптимизацию методом наименьших квадратов
    result = least_squares(
        reprojection_residual,  # Целевая функция
        initial_params,  # Начальные параметры
        args=(num_frames, num_points, K, observations),  # Доп. аргументы
        method='lm'  # Метод Левенберга-Марквардта
    )

    # Извлекаем оптимизированные параметры
    optimized_params = result.x
    optimized_poses = {}  # Словарь для хранения оптимизированных поз
    optimized_map_points = optimized_params[num_frames * 6:].reshape((num_points, 3))

    # Перебираем и извлекаем оптимизированные позы из параметров
    for i in range(num_frames):
        # Получаем оптимизированные rvec и tvec для каждой позы
        rvec = optimized_params[i * 6:i * 6 + 3]
        tvec = optimized_params[i * 6 + 3:i * 6 + 6]
        # Преобразуем вектор поворота в матрицу
        R, _ = cv2.Rodrigues(rvec)
        # Создаем матрицу позы и добавляем ее в словарь оптимизированных поз
        optimized_poses[i] = np.hstack((R, tvec.reshape(3, 1)))

    # Возвращаем оптимизированные позы и 3D-точки
    return optimized_poses, optimized_map_points
