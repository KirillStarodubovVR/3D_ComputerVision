import gtsam
import gtsam.noiseModel
import numpy as np

def bundle_adjustment(poses, points, features, observations, reprojection_threshold=5e-3):
    #bundle adjustment using gtsam
    #create factor graph
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()
    fix_first = False
    #add camera poses
    for i, pose in enumerate(poses):
        if pose is None:
            continue
        pose = gtsam.Pose3(gtsam.Rot3(pose[:3, :3]), gtsam.Point3(pose[:3, 3]))
        initial_estimate.insert(gtsam.symbol('p', i), pose.inverse())
        if fix_first:
            graph.add(gtsam.PriorFactorPose3(gtsam.symbol('p', i), gtsam.Pose3(), gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]))))
            fix_first = False
        

    #add 3d points
    for i, point in enumerate(points):
        point = gtsam.Point3(point)
        initial_estimate.insert(gtsam.symbol('l', i), point)

    #add reprojection factors
    #TODO:  добавьте все наблюдения в GT-SAM граф
    # вам нужно добавить gtsam.GenericProjectionFactorCal3_S2 для каждого наблюдения
    # каждый фактор принимает следующие аргументы:
    # 1. Наблюдаемая точка в виде gtsam.Point2
    # 2. Модель шума
    # 3. Символ камеры, например gtsam.symbol('p', camera_id)
    # 4. Символ точки, например gtsam.symbol('l', point_id)
    # 5. Модель камеры, как gtsam.Cal3_S2(fx, fy, s, u0, v0)
    # где fx и fy - фокусные расстояния, s - skew, u0 и v0 - координаты центра проекции

    noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
    huber = gtsam.noiseModel.mEstimator.Huber(reprojection_threshold)
    noise = gtsam.noiseModel.Robust(huber, noise)


    #optimize
    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosityLM("SUMMARY")
    params.setlambdaInitial(1e-3)
    params.setMaxIterations(10)
    params.setlambdaUpperBound(1e7)
    params.setlambdaLowerBound(1e-7)
    params.setRelativeErrorTol(1e-5)

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()

    #update poses and points
    for i in range(len(poses)):
        if poses[i] is not None:
            poses[i] = np.eye(4) 
            pose = result.atPose3(gtsam.symbol('p', i)).inverse()
            poses[i][:3, :3] = pose.rotation().matrix()
            poses[i][:3, 3] = pose.translation()
        
    for i in range(len(points)):
        point = result.atPoint3((gtsam.symbol('l', i)))
        points[i] = point


# от ChatGPT
"""
# Импортируем необходимые библиотеки для выполнения bundle adjustment
import gtsam
import gtsam.noiseModel
import numpy as np

# Определяем функцию bundle_adjustment, которая выполняет настройку пучка (связка камеры и точек)
def bundle_adjustment(poses, points, features, observations, reprojection_threshold=5e-3):
    # Инициализация bundle adjustment с использованием gtsam
    # Создаем факторный граф
    graph = gtsam.NonlinearFactorGraph()
    # Инициализируем начальное приближение значений поз камеры и точек
    initial_estimate = gtsam.Values()
    # Флаг для фиксации первой позы камеры
    fix_first = False
    
    # Добавляем позы камер в начальное приближение
    for i, pose in enumerate(poses):
        # Пропускаем, если pose равна None
        if pose is None:
            continue
        # Преобразуем матрицу позы в объект Pose3
        pose = gtsam.Pose3(gtsam.Rot3(pose[:3, :3]), gtsam.Point3(pose[:3, 3]))
        # Вставляем инвертированную позу в начальное приближение с символом камеры
        initial_estimate.insert(gtsam.symbol('p', i), pose.inverse())
        
        # Если нужно зафиксировать первую камеру, добавляем Prior фактор для неё
        if fix_first:
            # Задаем приоритетную позу с малой ошибкой для всех шести параметров
            graph.add(gtsam.PriorFactorPose3(
                gtsam.symbol('p', i), 
                gtsam.Pose3(), 
                gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-3] * 6))
            ))
            fix_first = False  # Отключаем фиксацию после первой камеры

    # Добавляем 3D точки в начальное приближение
    for i, point in enumerate(points):
        # Преобразуем точку в формат gtsam.Point3
        point = gtsam.Point3(point)
        # Вставляем точку с уникальным символом
        initial_estimate.insert(gtsam.symbol('l', i), point)

    # Добавляем факторы репроекции (переход из 3D в 2D) в граф
    # TODO: добавить все наблюдения в GT-SAM граф
    # Для этого нужно добавить gtsam.GenericProjectionFactorCal3_S2 для каждого наблюдения,
    # каждый фактор принимает аргументы: точка, модель шума, символы камеры и точки, модель камеры

    # Определяем изотропный шум на основе отклонения в 1 пиксель
    noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
    # Применяем метод Хьюбера для уменьшения влияния выбросов
    huber = gtsam.noiseModel.mEstimator.Huber(reprojection_threshold)
    # Создаем комбинированную модель шума с робастностью
    noise = gtsam.noiseModel.Robust(huber, noise)

    # Настраиваем параметры оптимизатора Левенберга-Марквардта
    params = gtsam.LevenbergMarquardtParams()
    # Устанавливаем режим вывода резюме
    params.setVerbosityLM("SUMMARY")
    # Начальное значение параметра λ для баланса
    params.setlambdaInitial(1e-3)
    # Максимальное количество итераций для оптимизации
    params.setMaxIterations(10)
    # Верхний и нижний пределы для параметра λ
    params.setlambdaUpperBound(1e7)
    params.setlambdaLowerBound(1e-7)
    # Устанавливаем допуск для относительной ошибки
    params.setRelativeErrorTol(1e-5)

    # Запускаем оптимизацию с настройками и графом
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()

    # Обновляем позы камер на основе оптимизированных значений
    for i in range(len(poses)):
        if poses[i] is not None:
            # Создаем единичную матрицу 4x4 для позы
            poses[i] = np.eye(4)
            # Получаем обратную позу из результата оптимизации
            pose = result.atPose3(gtsam.symbol('p', i)).inverse()
            # Обновляем матрицу вращения
            poses[i][:3, :3] = pose.rotation().matrix()
            # Обновляем вектор трансляции
            poses[i][:3, 3] = pose.translation()
    
    # Обновляем точки на основе оптимизированных значений
    for i in range(len(points)):
        # Получаем оптимизированную точку из результата
        point = result.atPoint3(gtsam.symbol('l', i))
        # Обновляем координаты точки
        points[i] = point
"""