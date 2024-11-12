import numpy as np
import cv2

def solve_pnp(points, landmark_ids, keypoints, keypoints_ids, reprojection_threshold=5e-3):
    # Решаем задачу PnP (вычисление положения камеры) с использованием 3D-точек и 2D-ключевых точек
    # Получаем 3D-точки, соответствующие landmark_ids (идентификаторы ориентиров)
    points = points[landmark_ids]
    # Получаем 2D-ключевые точки, соответствующие keypoints_ids (идентификаторы ключевых точек)
    keypoints = np.array([keypoints[keypoints_id] for keypoints_id in keypoints_ids])

    # Запускаем алгоритм cv2.solvePnPRansac для нахождения положения камеры
    # rvec и tvec - вектор вращения и вектор перемещения, inliers - список инлайеров (соответствий, которые прошли проверку)
    _, rvec, tvec, inliers = cv2.solvePnPRansac(points,
                                                keypoints,
                                                np.eye(3),
                                                np.zeros(5),
                                                confidence=0.99,
                                                reprojectionError=reprojection_threshold,
                                                iterationsCount=1000)

    # Выводим количество найденных инлайнеров
    if inliers is None or len(inliers) < 100:
        print('No inliers found') # Сообщаем, что инлайнеров не найдено
        return None, None, None

    # Выводим количество найденных инлайнеров
    print('Found {} inliers'.format(len(inliers)))
    # Преобразуем вектор вращения в матрицу вращения R с помощью cv2.Rodrigues
    R, _ = cv2.Rodrigues(rvec)
    # Объединяем матрицу вращения и вектор перемещения в матрицу трансформации T
    T = np.hstack((R, tvec))

    # Фильтруем индексы landmark_ids и keypoints_ids, оставляя только инлайнеры
    landmark_ids = np.array(landmark_ids)[inliers.flatten()]
    keypoints_ids = np.array(keypoints_ids)[inliers.flatten()]

    # Возвращаем матрицу трансформации T, а также обновленные списки landmark_ids и keypoints_ids
    return T, landmark_ids, keypoints_ids
