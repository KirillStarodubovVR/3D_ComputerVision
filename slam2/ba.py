import gtsam
import gtsam.noiseModel
import numpy as np

def bundle_adjustment(poses, points, features, observations, reprojection_threshold=5e-3):
    # Выполняет совместное уточнение параметров камеры и 3D-точек (bundle adjustment) с использованием библиотеки GTSAM
    # Создаем факторный граф, который будет содержать факторы для оптимизации
    graph = gtsam.NonlinearFactorGraph()
    # Создаем начальные оценки для поз камеры и положения 3D-точек
    initial_estimate = gtsam.Values()
    # Флаг для фиксации первой камеры
    fix_first = False

    # Добавляем позы камер в начальную оценку
    for i, pose in enumerate(poses):
        # Пропускаем камеры без начальной позы
        if pose is None:
            continue
        # Преобразуем матрицу позы в формат GTSAM
        pose = gtsam.Pose3(gtsam.Rot3(pose[:3, :3]), gtsam.Point3(pose[:3, 3]))
        # Вставляем инвертированную позу камеры в начальные значения
        initial_estimate.insert(gtsam.symbol('p', i), pose.inverse())
        # Фиксируем первую камеру, чтобы избежать неопределенности в положении
        if fix_first:
            graph.add(gtsam.PriorFactorPose3(gtsam.symbol('p', i),
                                             gtsam.Pose3(),
                                             gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]))))
            fix_first = False

    # Добавляем 3D-точки в начальные значения
    for i, point in enumerate(points):
        point = gtsam.Point3(point)  # Преобразуем точку в формат GTSAM
        initial_estimate.insert(gtsam.symbol('l', i), point)   # Добавляем в начальную оценку

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

    # Параметры камеры, используемые для всех наблюдений
    # fx, fy, s, u0, v0 = 1280.0, 1280.0, 0.0, 768.0, 576.0
    fx, fy, s, u0, v0 = 1.0, 1.0, 0.0, 0.0, 0.0
    K = gtsam.Cal3_S2(fx, fy, s, u0, v0) # Модель камеры

    # Задаем шум для факторов проекции и создаем модель Huber'а для более устойчивой оптимизации
    noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0) # Изотропная модель шума
    huber = gtsam.noiseModel.mEstimator.Huber(reprojection_threshold)  # Устанавливаем порог для Huber'а
    noise = gtsam.noiseModel.Robust(huber, noise)  # Создаем робастную модель шума

    for point_id, obs in enumerate(observations):
        for (camera_id, feature_id) in obs:
            observed_point = features[camera_id][0][feature_id]  # Получаем координаты 2D-наблюдения
            # Добавляем фактор проекции с использованием модели камеры
            graph.add(gtsam.GenericProjectionFactorCal3_S2(
                gtsam.Point2(observed_point),  # Наблюдаемая точка
                noise,  # Модель шума
                gtsam.symbol('p', camera_id),  # Символ камеры
                gtsam.symbol('l', point_id),  # Символ точки
                K  # Модель камеры
            ))


    # Параметры оптимизации для алгоритма Левенберга-Марквардта
    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosityLM("SUMMARY") # Уровень детализации вывода
    params.setlambdaInitial(1e-3)  # Начальное значение параметра лямбда
    params.setMaxIterations(10)  # Максимальное количество итераций
    params.setlambdaUpperBound(1e7)  # Верхний предел лямбда
    params.setlambdaLowerBound(1e-7)  # Нижний предел лямбда
    params.setRelativeErrorTol(1e-5)  # Относительный порог ошибки для остановки

    # Создаем оптимизатор Левенберга-Марквардта и выполняем оптимизацию
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()

    # Обновляем позы камер с результатами оптимизации
    for i in range(len(poses)):
        if poses[i] is not None:
            poses[i] = np.eye(4)  # Создаем новую матрицу для позы
            pose = result.atPose3(gtsam.symbol('p', i)).inverse()  # Извлекаем инвертированную позу камеры
            poses[i][:3, :3] = pose.rotation().matrix()  # Записываем матрицу вращения
            poses[i][:3, 3] = pose.translation()  # Записываем вектор трансляции

    # Обновляем положения 3D-точек с результатами оптимизации
    for i in range(len(points)):
        point = result.atPoint3((gtsam.symbol('l', i)))  # Извлекаем позицию точки
        points[i] = point  # Обновляем точку
