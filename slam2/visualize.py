import open3d as o3d
import numpy as np

def draw_scene(poses, pointcloud, camera_model):
    WIDTH = 1280
    HEIGHT = 720

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(width=WIDTH, height=HEIGHT)

    for pose in poses:
        if pose is not None:
            camera_obj = visualizer.get_view_control().convert_to_pinhole_camera_parameters()
            camera_obj.intrinsic.set_intrinsics(camera_model.size[0], camera_model.size[1], camera_model.focal_length, camera_model.focal_length, camera_model.size[0]//2, camera_model.size[1]//2)
            extrinsic = np.eye(4)
            extrinsic[:3, :] = pose[:3, :]
            camera_lines = o3d.geometry.LineSet.create_camera_visualization(view_width_px=camera_model.size[0], view_height_px=camera_model.size[1], intrinsic=camera_obj.intrinsic.intrinsic_matrix, extrinsic=extrinsic, scale = 0.2)
            visualizer.add_geometry(camera_lines)
        
    #add points
    points = o3d.geometry.PointCloud()
    points.points = o3d.utility.Vector3dVector(pointcloud)
    colors = np.zeros((len(pointcloud), 3))
    points.colors = o3d.utility.Vector3dVector(colors)
        
    visualizer.add_geometry(points)
    visualizer.run()

# ChatGPT

"""
import open3d as o3d  # Импортируем Open3D для работы с 3D-объектами и визуализацией
import numpy as np  # Импортируем numpy для работы с массивами и линейной алгеброй

# Функция для визуализации сцены с камерами и точечным облаком
def draw_scene(poses, pointcloud, camera_model):
    # Устанавливаем размеры окна визуализации
    WIDTH = 1280
    HEIGHT = 720

    # Создаем объект визуализатора
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(width=WIDTH, height=HEIGHT)  # Создаем окно с заданными размерами

    # Проходим по всем позициям камер
    for pose in poses:
        if pose is not None:  # Если позиция камеры не пустая
            # Получаем параметры камеры из объекта визуализации
            camera_obj = visualizer.get_view_control().convert_to_pinhole_camera_parameters()
            # Задаем параметры интринсик модели камеры (внутренние параметры)
            camera_obj.intrinsic.set_intrinsics(camera_model.size[0], camera_model.size[1], camera_model.focal_length, camera_model.focal_length, camera_model.size[0]//2, camera_model.size[1]//2)
            # Создаем матрицу экструзии для преобразования в мировые координаты
            extrinsic = np.eye(4)
            extrinsic[:3, :] = pose[:3, :]  # Копируем матрицу поворота и трансляции
            # Создаем объект визуализации камеры как набор линий
            camera_lines = o3d.geometry.LineSet.create_camera_visualization(
                view_width_px=camera_model.size[0], view_height_px=camera_model.size[1],
                intrinsic=camera_obj.intrinsic.intrinsic_matrix, extrinsic=extrinsic, scale=0.2
            )
            # Добавляем объект камеры в визуализатор
            visualizer.add_geometry(camera_lines)
        
    # Создаем и добавляем точечное облако
    points = o3d.geometry.PointCloud()
    points.points = o3d.utility.Vector3dVector(pointcloud)  # Устанавливаем точки из массива pointcloud
    colors = np.zeros((len(pointcloud), 3))  # Создаем массив цветов (все точки черные)
    points.colors = o3d.utility.Vector3dVector(colors)  # Присваиваем цвета точечному облаку

    # Добавляем точечное облако в визуализатор
    visualizer.add_geometry(points)
    # Запускаем визуализацию
    visualizer.run()


"""