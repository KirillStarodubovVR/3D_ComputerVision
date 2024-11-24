# Импорт библиотек и необходимых сервисных функций
import numpy as np
# from Scripts.jsonpointer import parse_pointer
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from torch import dtype

from cloud_utils import PointCloud,plot_pc_data3d,plot_bboxes_3d,get_voxel_corners,PCD_SCENE
import open3d as o3d
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle


# Загрузка данных
file = "data/000021.bin"
cloud = PointCloud.from_file(str(file))
points = cloud.points

#Визуализируем входное облако точек
#В качестве цветовой палитры используем значение интенсивности точек
# pc_plots = plot_pc_data3d(x=points[:,0], y=points[:,1], z=points[:,2],apply_color_gradient=True, color=points[:,3])
# # Если apply_color_gradient== True -> В качестве цветовой палитры будет использоваться дистанция точки от центра лидара
# layout = dict(template="plotly_dark", title="Raw Point cloud", scene=PCD_SCENE, title_x=0.5)
# fig = go.Figure(data=pc_plots, layout=layout)
# fig.show()


def cloud_to_bev_image(cloud, x_range=[-30, 30], y_range=[-30, 30], z_range=[-2, 2], cell_size=0.2):
    """
    cloud: [N, 4] Облако точек (x,y,z,intensity)
    x_range: границы по x от - до + метров
    y_range: границы по y от - до + метров
    z_range: границы по z от - до + метров
    cell_size : область, описываемая 1 пикселем на bev (kxk метров)
    """
    # Шаг 1. : Фильтрация данных согласно координатам
    mask = np.where((cloud[:, 0] >= x_range[0]) & (cloud[:, 0] <= x_range[1]) &
                    (cloud[:, 1] >= y_range[0]) & (cloud[:, 1] <= y_range[1]) &
                    (cloud[:, 2] >= z_range[0]) & (cloud[:, 2] <= z_range[1]))
    cloud = cloud[mask]

    # Шаг 2. Перевод координат в пространство изображения
    cloud_cpy = np.copy(cloud)
    cloud_cpy[:, 0] = (np.floor((x_range[1] - cloud_cpy[:, 0]) / cell_size)).astype(int)
    cloud_cpy[:, 1] = (np.floor((y_range[1] - cloud_cpy[:, 1]) / cell_size)).astype(int)

    # Шаг 2.1 Смещение z (чтобы убрать значения меньше 0)
    cloud_cpy[:, 2] = cloud_cpy[:, 2] - z_range[0]

    # Шаг 3. Заполнение intensity слоя
    height = np.int_((x_range[1] - x_range[0]) / cell_size)
    width = np.int_((y_range[1] - y_range[0]) / cell_size)

    intensity_map = np.zeros((height, width), dtype=np.float32)
    mean_intensity_map = np.zeros((height, width), dtype=np.float32)
    intensity_count = np.zeros((height, width), dtype=np.uint32)
    for i in range(cloud_cpy.shape[0]):
        x, y, intensity = int(cloud_cpy[i, 0]), int(cloud_cpy[i, 1]), cloud_cpy[i, 3]
        intensity_map[x, y] += intensity
        intensity_count[x, y] += 1

    mask = intensity_count > 0
    mean_intensity_map[mask] = intensity_map[mask] / intensity_count[mask]

    # Шаг 4. Заполнение max height слоя
    height_map = np.zeros((height, width))
    for i in range(cloud_cpy.shape[0]):
        x, y, z = int(cloud_cpy[i, 0]), int(cloud_cpy[i, 1]), cloud_cpy[i, 2]
        height_map[x, y] = max(height_map[x, y], z)

    # Шаг 5. Заполнение density слоя
    density_map = np.zeros((height, width))
    point_counts = np.zeros((height, width))
    for i in range(cloud_cpy.shape[0]):
        x, y = int(cloud_cpy[i, 0]), int(cloud_cpy[i, 1])
        density_map[x, y] += 1

    # Шаг 6. Нормализация
    intensity_map = mean_intensity_map / np.max(mean_intensity_map)
    height_map = height_map / np.max(height_map)
    density_map = density_map / np.max(density_map)

    # Раскоментируйте для проверки результата
    # np.testing.assert_array_equal(intensity_map, np.loadtxt("asserts/task_1/intensity_f.txt"))
    # np.testing.assert_array_equal(density_map, np.loadtxt("asserts/task_1/density_f.txt"))
    # np.testing.assert_array_equal(height_map, np.loadtxt("asserts/task_1/height_f.txt"))

    # Шаг 7. Формирование итогового 3-канального изображения
    bev_map = np.zeros((height, width, 3))
    bev_map[:, :, 0] = intensity_map[:height, :width]
    bev_map[:, :, 1] = height_map[:height, :width]
    bev_map[:, :, 2] = density_map[:height, :width]
    bev_map = (bev_map * 255).astype(np.uint8)

    return bev_map

# bev_map=cloud_to_bev_image(points)
# plt.figure(figsize = (20,10))
# plt.imshow(bev_map)
# plt.show()


def cloud_to_range_image(cloud, h, w, vertical_angle_up, vertical_angle_down):
    """
    cloud: [N, 4] Облако точек (x,y,z,intensity)
    h : высота для range view image
    w : ширина для range view image
    vertical_angle_up : ограничение по вертикальному углу сверху
    vertical_angle_down : ограничение по вертикальному углу снизу
    """
    # Шаг 1. - Сконвертируем координаты в сферическую систему
    r = np.sqrt(cloud[:, 0]**2 + cloud[:, 1]**2 + cloud[:, 2]**2)
    phi = np.arctan2(cloud[:, 1], cloud[:, 0])
    theta = np.arcsin(cloud[:, 2] / r)
    height = cloud[:, 2]
    intensity = cloud[:, 3]

    # Шаг 2. - Проинициализируем константы (переводим в радианы)
    fov_up = vertical_angle_up / 180.0 * np.pi
    fov_down = vertical_angle_down / 180.0 * np.pi

    # Шаг 2.1 - Проинициализируем итоговые представления
    range_map = np.zeros((h, w))
    intensity_map = np.zeros((h, w))
    height_map = np.zeros((h, w))
    angle_map = np.zeros((h, w))
    occupancy_map = np.zeros((h, w))

    # Шаг 3. - Найдем индексы по горизонтали
    proj_x = np.floor(0.5 * (1 + phi / np.pi) * w).astype(np.int32)

    # Шаг 4. - Найдем индексы по вертикали
    proj_y = np.floor((fov_up - theta) / (fov_up - fov_down) * h).astype(np.int32)

    # Шаг 5. - Приведем найденные индексы к границам [0, H-1], [0, W-1]
    proj_x = np.clip(proj_x, 0, w - 1)
    proj_y = np.clip(proj_y, 0, h - 1)

    # Разкоментируйте для проверки результата
    np.testing.assert_array_equal(proj_x,np.loadtxt("asserts/task_2/proj_x.txt"))
    np.testing.assert_array_equal(proj_y,np.loadtxt("asserts/task_2/proj_y.txt"))

    # Шаг 6. - Упорядочим все элементы согласно уменьшению r
    order = np.argsort(-r)  # Сортировка по убыванию r
    r = r[order]
    phi = phi[order]
    theta = theta[order]
    height = height[order]
    intensity = intensity[order]
    proj_x = proj_x[order]
    proj_y = proj_y[order]

    # Шаг 7. - Заполним итоговые представления значениями r, intensity, height, theta, occupancy
    range_map[proj_y, proj_x] = r
    # Заполняем карту дальности
    intensity_map[proj_y, proj_x] = intensity
    # Заполняем карту интенсивности
    height_map[proj_y, proj_x] = height
    # Заполняем карту высот
    angle_map[proj_y, proj_x] = np.degrees(phi)
    # Конвертируем горизонтальные углы в градусы
    occupancy_map[proj_y, proj_x] = 1
    # Устанавливаем бинарное значение занятости

    # Шаг 8. - Нормализация от 0 до 1
    range_map = range_map / np.max(range_map)
    # Нормализуем карту дальности
    intensity_map = intensity_map / np.max(intensity_map)
    # Нормализуем карту интенсивности
    height_map = height_map - np.min(height_map)
    height_map = height_map / np.max(height_map)
    # Нормализуем карту высот
    angle_map = angle_map - np.min(angle_map)
    angle_map = angle_map / np.max(angle_map)
    # Нормализуем карту углов

    # Шаг 9. - Заполним финальное представление
    # Финальная карта с 5 каналами: дальность, интенсивность, высота, угол, занятость
    range_view_map = np.zeros((h, w, 5))
    # Канал занятости
    range_view_map[:, :, 4] = occupancy_map[:h, :w]
    # Канал углов
    range_view_map[:, :, 3] = angle_map[:h, :w]
    # Канал высот
    range_view_map[:, :, 2] = height_map[:h, :w]
    # Канал интенсивности
    range_view_map[:, :, 1] = intensity_map[:h, :w]
    # Канал дальности
    range_view_map[:, :, 0] = range_map[:h, :w]
    # Преобразуем значения в диапазон от 0 до 255 для визуализации
    range_view_map = (range_view_map * 255).astype(np.uint8)

    # Возвращаем финальную карту
    return range_view_map

def furthest_point_sampling(points, n_samples):
    """
    points: [N, 3] array containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically << N
    """
    # Шаг 1. - Скопируем оригинальное облако точек
    points_cpy = np.copy(points)
    # Шаг 2. - Проинициализируем индексы для точек которые не были выбраны
    points_left = np.arange(len(points))
    # Шаг 3. - Проинициализируем индексы для того количества точек которое мы хотим просемплировать
    sample_inds = np.zeros(n_samples, dtype='int')
    # Шаг 4. - Проинициализируем дистанции для точек
    dists = np.ones(len(points)) * float('inf')
    # Шаг 5. - Выберем случайную точку, сохраним ее индекс в sample_inds и удалим ее из массива points_left
    selected = np.random.choice(points_left)
    sample_inds[0] = selected

    points_left = np.delete(points_left, np.where(points_left == selected))

    # Шаг 6. - Итеративный выбор точек и семплирование
    for i in range(1, n_samples):
        # Шаг 6.1 - Возьмем последнюю добавленную точку и подсчитаем L2 дистанцию от нее до не выбранных точек
        last_selected = points_cpy[sample_inds[i - 1]]
        dist_to_last_selected = np.linalg.norm(points_cpy[points_left] - last_selected, axis=1)

        # Шаг 6.2 - Обновим значение дистанций в dists для тех индексов, где дистанция больше дистанции до последней добавленной точки
        dists[points_left] = np.minimum(dists[points_left], dist_to_last_selected)

        # Шаг 6.3 - Выберем индекс точки с максимальной дистанцией записанной в dists
        next_selected_idx = np.argmax(dists[points_left])
        next_selected = points_left[next_selected_idx]

        # Шаг 6.4 - Запишем выбранный индекс точки в sample_inds
        sample_inds[i] = next_selected

        # Шаг 6.5 - Удалим выбранную точку из points_left
        points_left = np.delete(points_left, next_selected_idx)

    return points_cpy[sample_inds]

def furthest_point_sampling_mod(points, n_samples):
    """
    points: [N, 3] array containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically << N
    """
    # Шаг 1. - Скопируем оригинальное облако точек
    points_cpy = np.copy(points)
    # Шаг 2. - Проинициализируем индексы для точек которые не были выбраны

    kmeans = KMeans(n_clusters=n_samples, random_state=0, n_init="auto").fit(points)
    cluster_centers = kmeans.cluster_centers_

    sampled_points = []

    for center in cluster_centers:

        distances = np.linalg.norm(points_cpy - center, axis=1)
        min_dist_ind = np.argmin(distances)
        sampled_points.append(points_cpy[min_dist_ind])


    return np.array(sampled_points)

voxel_size = [1,1,0.2]
x_range = [-30,30]
y_range = [-30,30]
z_range = [-2,2]
num_points = 100
def make_voxel_encoding(cloud,
                        x_range=x_range,
                        y_range=y_range,
                        z_range=z_range,
                        voxel_size=voxel_size,
                        max_point_number = num_points):

    with open("asserts/task_4/voxel_features.pkl", 'rb') as file:
        ttt = np.array(pickle.load(file))

    # Шаг 1. Сдвинем все координаты в диапазон [0,inf)
    shifted_cloud = np.copy(cloud)
    shifted_cloud[:,:3] = shifted_cloud[:,:3] - [x_range[0],y_range[0],z_range[0]]

    # Шаг 2. Определим размеры итоговой координатной сетки
    grid_size = [(x_range[1] - x_range[0])/voxel_size[0],
                 (y_range[1] - y_range[0])/voxel_size[1],
                 (z_range[1] - z_range[0])/voxel_size[2]]

    # Шаг 3. Найдем координаты вокселя для каждой точки

    # voxel_index = np.floor(shifted_cloud[:,:3] / voxel_size).astype(int) # Формат int64
    voxel_index = np.floor(shifted_cloud[:,:3] / voxel_size) # Формат float64

    # Разкоментируйте для проверки результата
    np.testing.assert_array_equal(voxel_index,np.loadtxt("asserts/task_4/voxel_index.txt"))

    # Шаг 4. Отфильтруем воксели выходящие за пределы координатной сетки
    # 0 <= x <= 60; 0 <= y <= 60; 0 <= z <= 20
    valid_mask = (
        (voxel_index[:, 0] >= 0) & (voxel_index[:, 0] < grid_size[0]) &
        (voxel_index[:, 1] >= 0) & (voxel_index[:, 1] < grid_size[1]) &
        (voxel_index[:, 2] >= 0) & (voxel_index[:, 2] < grid_size[2])
    )
    # Фильтруем воксели по маске
    voxel_index = voxel_index[valid_mask]
    # Фильтруем облако точек
    shifted_cloud = shifted_cloud[valid_mask]

    # Шаг 5. Проинициализируем буфферы
    unique, unique_indices, unique_inverse, unique_counts = np.unique(voxel_index,
                                                                      axis=0,
                                                                      return_index=True,
                                                                      return_counts=True,
                                                                      return_inverse=True)
    # np.where((voxel_index == unique[0]).all(axis=1))
    # Шаг 5.1 Уникальные индексы вокселей - [K,3]
    voxel_indexes = unique # sorted x->y->z
    K = voxel_indexes.shape[0]

    # Шаг 5.2 Количество точек в вокселе - [K,1]
    points_count = np.zeros((K, ), dtype=np.int32)

    # Шаг 5.3 Признаки для каждой из точек в вокселе - [K, T, 7]
    voxel_features = np.zeros((K, max_point_number, 7), dtype=np.float32) # проверить тип данных

    # Шаг 6. Заполним буфферы данными
    # for ind, point in zip(unique_inverse, shifted_cloud):
    #     num_point = points_count[ind]
    #     if num_point < max_point_number:
    #         voxel_features[ind, num_point, :4] = point
    #         points_count[ind] += 1

    for ind, point in zip(unique_inverse, shifted_cloud):
        num_point = points_count[ind]
        if num_point < max_point_number:
            voxel_features[ind, num_point, :4] = point
            points_count[ind] += 1


    # Шаг 7. Для буффера признаков вычислим представление точки согласно формуле (Шаг 7)
    # for ind in range(K):
    #     pts_in_voxel = voxel_features[ind, :points_count[ind], :3]
    #     # стоит ли учитывать нулевое значение для расчёта центройда
    #     centroid = pts_in_voxel.mean(axis=0)
    #     voxel_features[ind, :points_count[ind], 4:7] = pts_in_voxel - centroid

    for ind in range(K):
        pts_in_voxel = voxel_features[ind, :points_count[ind], :3]
        # возвращаем к исходной СК
        pts_in_voxel = pts_in_voxel + [x_range[0],y_range[0],z_range[0]]
        voxel_features[ind, :points_count[ind], :3] = pts_in_voxel
        centroid = pts_in_voxel.mean(axis=0)
        pts_by_centroid = pts_in_voxel - centroid
        voxel_features[ind, :points_count[ind], 4:7] = pts_in_voxel - centroid
        voxel_features[ind, points_count[ind]:, 4:7] = -centroid


    # Раскоментируйте для проверки результата
    np.testing.assert_array_equal(voxel_indexes,pickle.load(open("asserts/task_4/voxel_indexes.pkl", 'rb')))
    # np.testing.assert_array_equal(voxel_features, pickle.load(open("asserts/task_4/voxel_features.pkl", 'rb')))
    np.testing.assert_array_equal(points_count,np.loadtxt("asserts/task_4/points_count.txt"))





    return voxel_features,voxel_indexes,points_count,grid_size




# h = 64
# w = 640
# vertical_angle_up = 5
# vertical_angle_down = -30
# range_view_map = cloud_to_range_image(points,h,w,vertical_angle_up,vertical_angle_down)
#
# plt.figure(figsize = (30,10))
# plt.imshow(range_view_map[:,:,0]) # канал range
# plt.show()


# n_samples = 2000
# fps_points = furthest_point_sampling(points,n_samples)
# #Визуализируем начальное облако точек
# pc_plots = plot_pc_data3d(x=points[:,0], y=points[:,1], z=points[:,2],apply_color_gradient=False, color=points[:,3])
# layout = dict(template="plotly_dark", title="Raw Point cloud", scene=PCD_SCENE, title_x=0.5)
# fig = go.Figure(data=pc_plots, layout=layout)
# fig.show()
#
# fps_pc_plots = plot_pc_data3d(x=fps_points[:,0], y=fps_points[:,1], z=fps_points[:,2],apply_color_gradient=False, color=points[:,3])
# layout = dict(template="plotly_dark", title="FPS Point cloud", scene=PCD_SCENE, title_x=0.5)
# fig = go.Figure(data=fps_pc_plots, layout=layout)
# fig.show()
#
# fps_points = furthest_point_sampling_mod(points,n_samples)
# fps_pc_plots = plot_pc_data3d(x=fps_points[:,0], y=fps_points[:,1], z=fps_points[:,2],apply_color_gradient=False, color=points[:,3])
# layout = dict(template="plotly_dark", title="FPS Point cloud", scene=PCD_SCENE, title_x=0.5)
# fig = go.Figure(data=fps_pc_plots, layout=layout)
# fig.show()

voxel_features,voxel_indexes,points_count,grid_size = make_voxel_encoding(points)

voxel_corners = get_voxel_corners(voxel_indexes, voxel_size=np.array(voxel_size), pc_range=np.array([x_range[0],y_range[0],z_range[0]]))
voxel_generator_plots = plot_bboxes_3d(voxel_corners, box_colors=['white'] * len(voxel_corners))

fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]])
fig.update_layout(template="plotly_dark", scene=PCD_SCENE, scene2=PCD_SCENE, height = 400, width = 1000,
                title=f"VOXEL GENERATOR", title_x=0.5, title_y=0.95, margin=dict(r=0, b=0, l=0, t=0))

fig.add_trace(pc_plots, row=1, col=1)
for trace in voxel_generator_plots:
    fig.add_trace(trace, row=1, col=2)
fig.show()