import argparse
import os 
import cv2
import numpy as np

from camera_model import parse_camera_model
from images import get_image_files, load_images
from features import extract_features, match_features, cross_check
from two_view_geometry import two_view_geometry, triangulate_points
from visualize import *
from pnp import solve_pnp
from ba import bundle_adjustment
from data_plot import find_initial_pair_plot


class SFM:
    # Конструктор класса SFM, инициализирующий начальные данные
    def __init__(self,
                 features,
                 matches,
                 camera_model,
                 reprojection_threshold):
        # Сохраняем входные данные
        self.features = features  # Признаки изображений
        self.matches = matches  # Соответствия между изображениями
        self.camera_model = camera_model  # Модель камеры
        self.reprojection_threshold = reprojection_threshold  # Порог репроекции

        # Инициализация сцены
        # Позиции камер
        self.poses = [None for _ in range(len(self.features))]
        # Набор 3D-точек
        self.points = []
        # Наблюдения 3D-точек
        # observations of 3d points, observations[i] is a list of tuples (j, k) where j is the index of the image and k is the index of the keypoint in the image
        self.observations = []
        # Таблица соответствий для наблюдений
        # observations lookup table, for each image and keypoint index, it stores the index of the 3d point
        self.observations_lookup = {}
        # Флаг, сигнализирующий об ошибке в реконструкции
        self.failed = False
    
    def is_finished(self):
        # Проверка завершения процесса реконструкции
        return self.failed or all([pose is not None for pose in self.poses])
    
    def find_initial_pair(self):
        # Находим пару изображений с наибольшим количеством соответствий для начальной инициализации
        max_i = -1
        max_j = -1
        find_initial_pair_plot(self.matches)
        #TODO: Выберите пару изображений с которой вы хотите начать реконструкцию.
        # Верните индексы этих изображений
        total_num = 0
        for i in range(self.matches.shape[0]):
            for j in range(self.matches.shape[1]):
                if i != j:
                    # Подразумевается что между одинаковыми кадрами нет общих точек
                    cur_num = len(matches[i, j])
                    if cur_num > total_num:
                        total_num = cur_num
                        max_i = i
                        max_j = j
        print(f"Инициализированы изображения с индексами: {max_i} {max_j}")
        print(f"Отобраны по критерию большего значения совместных точек")
        return max_i, max_j

    def initialize(self):
        # Инициализация начальной пары изображений и построение 3D-точек
        #find two images with most matches
        max_i, max_j = self.find_initial_pair()
        T, points, kp_id1, kp_id2 = two_view_geometry(self.features[max_i][0],
                                                      self.features[max_j][0],
                                                      self.matches[max_i][max_j],
                                                      self.reprojection_threshold)

        self.poses[max_i] = np.eye(4)  # Положение первой камеры
        self.poses[max_j] = T  # Положение второй камеры
        self.points = points  # Инициализируем 3D-точки
        
        # Добавляем наблюдения 3D-точек
        self.observations = []
        # self.observations = [[(max_i, i), (max_j, j)] for i, j in zip(kp_id1, kp_id2)]
        for i, j in zip(kp_id1, kp_id2):
            self.observations.append([(max_i, i), (max_j, j)])


        # Добавляем наблюдения в таблицу соответствий
        for i, obs in enumerate(self.observations):
            for j in obs:
                assert(self.observations_lookup.get(j) is None)
                self.observations_lookup[j] = i
        print ("Intialized with {} points".format(len(points)))

    def sort_views(self):
        # Сортируем виды по количеству наблюдений для 3D-точек
        views = []
        # TODO: Отсортируйте изображения по вероятности того, что они успешно добавятся в сцену.
        # Верните список пар (индекс изображения, ваша метрика). Метрика может быть любой.
        # В таком порядке вы будете пробовать добавлять изображения в сцену.
        data = np.zeros_like(self.matches, dtype=np.int16)
        data_for_sort = []
        # Преобразуем данные в таблицу
        for i in range(matches.shape[0]):
            for j in range(i, matches.shape[1]):
                if i != j:
                    data[i, j] = len(matches[i, j])
                    data_for_sort.append((i, j, data[i, j]))

        data_for_sort = sorted(data_for_sort, key=lambda  x: x[2], reverse=True)

        available_images = list(range(len(self.poses)))
        # available_images = np.arange(len(self.poses))
        used_images = []
        # Удаляем позы (изображения) которые использовали
        for i in range(len(self.poses)):
            if self.poses[i] is not None:
                available_images.remove(i)
                used_images.append(i)

        for i, j, ck in data_for_sort:

            if i not in used_images:
                used_images.append(i)
                available_images.remove(i)
                views.append((i, ck))
            if j not in used_images:
                used_images.append(j)
                available_images.remove(j)
                views.append((j, ck))

        views.sort(key=lambda x: x[1], reverse=True)
        return views

    def add_view(self, i):
        # Добавляем новый вид (изображение) в сцену
        print (f"Adding view {i}")
        # Инициализируем множество совпадений для хранения общих 3D-точек
        matches = set()

        # TODO: Найдите общие точки между изображением i и точками, которые уже есть в сцене.
        # Чтобы сделать это, вам нужно найти все точки, которые были сопоставлены с точками
        # изображения i в других изображениях. Затем, для каждой такой точки, найдите соответствующую
        # 3D точку, если она есть в сцене. Добавьте все такие точки в множество matches.
        # Matches - это множество кортежей (landmark_id, descriptor_id), где:
        # - landmark_id: индекс 3D-точки в self.points
        # - descriptor_id: индекс ключевой точки в изображении i
        for j in range(len(self.poses)):
            if i == j or self.poses[j] is None:
                continue
            # найти все точки между j-м и i-м изображениями.
            matches_ji = self.matches[j,i]
            # detect indecies which are on the scene from image j
            # pts_ind_j_image = [pt_ind for pts_pair in self.observations for (im, pt_ind) in pts_pair if im == j]
            # After we should sort indecies to find pts from image i which are the same with the image j
            # pts_ind_i_image = [(pt_j, pt_i) for (pt_j, pt_i) in matches_ji if pt_j in pts_ind_j_image]

            for match_ji in matches_ji:
                ind_j, ind_i = match_ji
                landmark_id = self.observations_lookup.get((j, ind_j))
                if landmark_id is not None:
                    matches.add((landmark_id, ind_i))


        landmark_ids = [match[0] for match in matches]
        descriptor_ids = [match[1] for match in matches]

        # Проверка достаточности общих точек для построения вида
        if len(landmark_ids) < 10:
            print('Not enough points in common')
            return False # Если точек меньше 10, завершаем функцию

        # Сообщаем о количестве найденных общих точек
        print('Found {} points in common'.format(len(landmark_ids)))
        # Используем алгоритм PnP для нахождения позы камеры (матрица T)
        T, landmark_ids, descriptor_ids  = solve_pnp(self.points,
                                                     landmark_ids,
                                                     self.features[i][0],
                                                     descriptor_ids,
                                                     self.reprojection_threshold)

        # Позиционируем камеру для текущего изображения
        if T is None:
            print ("Failed to add view")
            return False

        # Очищаем дескрипторы, сопоставленные с несколькими 3D-точками
        # Подсчитываем, сколько раз каждый дескриптор сопоставлен с 3D-точкой
        descriptor_count = {}
        for d in descriptor_ids:
            descriptor_count[d] = descriptor_count.get(d, 0) + 1
        # Удаляем дескрипторы, которые были сопоставлены с несколькими 3D-точками
        landmark_ids = [landmark_ids[i] for i, d in enumerate(descriptor_ids) if descriptor_count[d] == 1]
        descriptor_ids = [d for d in descriptor_ids if descriptor_count[d] == 1]

        # Устанавливаем позицию камеры для текущего изображения
        self.poses[i] = T

        # Добавляем наблюдения для текущего вида
        assert(self.check_observations_consisntency())  # Проверяем целостность наблюдений
        for l, d in zip(landmark_ids, descriptor_ids):
            assert(len(self.observations) > l)  # Убеждаемся, что индекс l существует в observations
            self.observations[l].append((i, d))  # Добавляем наблюдение для 3D-точки
            assert self.observations_lookup.get((i, d)) is None, (i, d)  # Добавляем наблюдение для 3D-точки
            self.observations_lookup[(i, d)] = l  # Обновляем observations_lookup для быстрого доступа
        
        assert(self.check_observations_consisntency()) # Проверка целостности наблюдений

        # Триангулируем новые точки для текущего вида, сравнивая его с другими
        for j in range(len(self.features)):
            if self.poses[j] is None or i == j:
                continue  # Пропускаем, если для изображения j нет позы или оно совпадает с i
            self.trinagulate_points(i, j)  # Триангулируем между изображениями i и j
            assert(self.check_observations_consisntency())  # Проверяем целостность наблюдений после триангуляции
        # Выводим информацию о текущем состоянии сцены
        print("Scene has {} points".format(len(self.points)))
        print("Scene has {} cameras".format(len([pose for pose in self.poses if pose is not None])))
        return True  # Успешное добавление вида
    
    def trinagulate_points(self, i, j):
        # Триангулируем точки между двумя видами (кадрами) i и j
        # Получаем совпадения между изображениями i и j
        matches = self.matches[i][j]
        # Если совпадений меньше 50, выходим из функции, так как недостаточно данных для триангуляции
        if len(matches) < 50:
            return

        # Инициализируем списки для хранения индексов дескрипторов, которые нужно триангулировать
        descritors_ids_i = []
        descritors_ids_j = []

        # Перебираем совпадения и отбираем только те, которые еще не имеют 3D-точек
        for match in matches:
            if self.observations_lookup.get((i, match[0])) is None and self.observations_lookup.get((j, match[1])) is None:
                # Добавляем индексы дескрипторов в списки для триангуляции
                descritors_ids_i.append(match[0])
                descritors_ids_j.append(match[1])

        # Извлекаем дескрипторы для выбранных совпадений
        descriptors_i = np.array([self.features[i][0][d] for d in descritors_ids_i])
        descriptors_j = np.array([self.features[j][0][d] for d in descritors_ids_j])
        # Проверяем, что у нас есть хотя бы 10 совпадающих дескрипторов, иначе выходим из функции
        if descriptors_i.shape[0] < 10:
            return
        # Выполняем триангуляцию точек, получая 3D-координаты и маску инлайеров
        points, inliers = triangulate_points(descriptors_i, descriptors_j,
                                             self.poses[i], self.poses[j],
                                             self.reprojection_threshold)
        # Оставляем только инлайеры среди триангулированных точек и соответствующих индексов дескрипторов
        points = points[inliers]
        descritors_ids_i = np.array(descritors_ids_i)[inliers]
        descritors_ids_j = np.array(descritors_ids_j)[inliers]
        # Проверяем, что у нас осталось хотя бы 10 точек после фильтрации, иначе выходим
        #if not enough points were triangulated probably something went wrong
        if len(points) < 10:
            return
        # Выводим количество успешно триангулированных точек между видами i и j
        print (f"Triangulated {len(points)} points from view {i} and {j}")

        # Добавляем новые 3D-точки в массив всех точек
        self.points = np.vstack((self.points, points))

        # Добавляем наблюдения для каждой пары триангулированных точек
        for d_i, d_j in zip(descritors_ids_i, descritors_ids_j):
            # Записываем пару индексов дескрипторов и кадров в список наблюдений
            self.observations.append([(i, d_i), (j, d_j)])
            # Проверяем, что запись в observations_lookup еще не существует
            assert(self.observations_lookup.get((i, d_i)) is None)
            assert(self.observations_lookup.get((j, d_j)) is None)
            # Добавляем индексы в observations_lookup для быстрого доступа
            self.observations_lookup[(i, d_i)] = len(self.observations) - 1
            self.observations_lookup[(j, d_j)] = len(self.observations) - 1

    def add_next_view(self):
        # Выбираем следующий вид для добавления в сцену
        views = self.sort_views() # Сортируем виды, чтобы определить порядок добавления
        for view in views:
            # Пытаемся добавить вид в сцену
            if self.add_view(view[0]):
                return # Успешное добавление вида, выходим из функции
        # Если ни один вид не удалось добавить, сообщаем об ошибке
        print("Failed to add a view")
        self.failed = True  # Устанавливаем флаг о неудаче
        

    def visualize(self):
        # Визуализируем текущую сцену с помощью заранее определенной функции отрисовки
        draw_scene(self.poses, self.points, self.camera_model)

    def check_observations_consisntency(self):
        # Проверяем согласованность наблюдений с точками и таблицей наблюдений

        for i, obs in enumerate(self.observations):
            for j in obs:
                # Проверяем, что индекс наблюдения в observations_lookup совпадает с текущим индексом
                if self.observations_lookup.get(j) != i:
                    print (f"Observation {j} does not match lookup table, expected {i} got {self.observations_lookup.get(j)}")
                    print(self.observations[i]) # Выводим ожидаемое наблюдение
                    print(self.observations[self.observations_lookup.get(j)]) # Выводим найденное наблюдение
                    return False # Если наблюдение не совпадает, возвращаем False

        # Проверяем, что все записи в observations_lookup согласуются с наблюдениями
        for i, obs in self.observations_lookup.items():
            if i not in self.observations[obs]:
                print (f"Lookup table {i} does not match observations, expected {obs} got {self.observations[obs]}")
                return False  # Если найдено несоответствие, возвращаем False

        # Проверяем, что количество точек и наблюдений совпадает
        if len(self.points) != len(self.observations):
            print("Number of points and observations do not match")
            return False # Если количество не совпадает, возвращаем False
        # Если все проверки пройдены, возвращаем True
        return True
    
    def bundle_adjustment(self):
        # Проводим оптимизацию всех поз и 3D-точек методом bundle adjustment
        bundle_adjustment(self.poses, self.points, self.features, self.observations, self.reprojection_threshold)
        # Фильтруем выбросы после оптимизации
        self.filter_outliers()
        # Проверяем согласованность наблюдений после фильтрации
        assert(self.check_observations_consisntency())

    def filter_outliers(self):
        # Фильтрация выбросов, проверяя ошибку проецирования
        observations_deleted = 0  # Считаем количество удаленных наблюдений
        for i, obs in enumerate(self.observations):
            # Перебираем все наблюдения для каждой 3D-точки
            for j in obs:
                point = self.points[i]  # Получаем координаты текущей 3D-точки
                pose = self.poses[j[0]]  # Получаем позу камеры для наблюдения
                # Применяем трансформацию к точке для перехода в систему координат изображения
                point = pose @ np.hstack((point, 1))  # Гомогенные координаты
                point = point / point[2]  # Нормализуем точку по оси Z
                # Вычисляем ошибку проекции (расстояние между проекцией и фактической позицией)
                error = np.linalg.norm(point[:2] - self.features[j[0]][0][j[1]])

                # Если ошибка превышает порог, удаляем наблюдение как выброс
                if error > self.reprojection_threshold:
                    self.observations_lookup.pop(j)  # Удаляем из таблицы наблюдений
                    self.observations[i].remove(j)  # Удаляем наблюдение из списка
                    observations_deleted += 1  # Увеличиваем счетчик удаленных наблюдений
        print(f"Filtered {observations_deleted} observations")

        # Удаляем точки, у которых меньше двух наблюдений
        to_delete = set()  # Множество для хранения индексов точек для удаления
        for i, obs in enumerate(self.observations):
            if len(obs) < 2:  # Если точка имеет менее двух наблюдений
                to_delete.add(i)  # Добавляем индекс в to_delete

        # Удаляем выбранные точки из массива 3D-точек
        self.points = np.delete(self.points, list(to_delete), axis=0)
        # Удаляем наблюдения для точек из to_delete
        self.observations = [obs for i, obs in enumerate(self.observations) if i not in to_delete]
        print(f"Filtered {len(to_delete)} points")

        # Перестраиваем таблицу наблюдений (observations_lookup) для согласованности данных
        self.observations_lookup = {}
        for i, obs in enumerate(self.observations):
            for j in obs:
                # Обновляем observations_lookup для каждого наблюдения
                self.observations_lookup[j] = i

        
# Основная функция программы
if __name__ == '__main__':
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Simple sfm pipeline')
    parser.add_argument('--input', type=str, help='input folder')
    parser.add_argument('--camera_model', type=str, help='camera model', default='SIMPLE_RADIAL 3072 2304 2559.68 1536 1152 -0.0204997')
    parser.add_argument('--max_views', type=int, help='maximum number of views', default=-1)
    parser.add_argument('--shuffle', action='store_true', help='shuffle images')
    parser.add_argument('--resize_factor', type=float, help='resize factor', default=1.0)
    parser.add_argument('--num_features', type=int, help='number of features', default=10000)
    parser.add_argument('--recalculate', action='store_true', help='recalculate features and matches')
    parser.add_argument('--reprojection_threshold', type=float, help='reprojection threshold', default=1e-3)
    parser.add_argument('--ba_frequency', type=int, help='bundle adjustment frequency', default=5)
    parser.add_argument('--vis_frequency', type=int, help='visualization frequency', default=5)
    args = parser.parse_args()

    # Added sort to get_image_files, turn it off!!!

    # Загрузка модели камеры
    camera_model = parse_camera_model(args.camera_model)
    if args.resize_factor != 1.0:
        camera_model.resize(args.resize_factor)
    print(camera_model)

    # Перепроверка необходимости пересчета фич и соответствий
    if not args.recalculate and (not os.path.exists('features.npy') or not os.path.exists('matches.npy')):
        args.recalculate = True # Если нет features and matches тогда устанавливаем флаг на пересчёт

    # Сохранение аргументов в файл
    with open('args.txt', 'w') as f:
        f.write(str(args) + '\n')
    #load images
    if args.recalculate:
        files = get_image_files(args.input, args.max_views, args.shuffle)
        #save image names
        with open('image_names.txt', 'w') as f:
            for file in files:
                f.write(file + '\n')
        images = load_images(files, args.resize_factor)
    else:
        files = [line.rstrip('\n') for line in open('image_names.txt')]
    

    #extract features
    if args.recalculate:
        features = extract_features(images, args.num_features)
        #unproject all features
        for i in range(len(features)):
            features[i][0] = camera_model.unproject(features[i][0])
        np.save('features.npy', features, allow_pickle=True)
    else:
        print("Loading features...")
        features = np.load('features.npy', allow_pickle=True)

    if args.recalculate:
        matches = match_features(features) # with base features.py
        matches = cross_check(matches) # with base features.py
        np.save('matches.npy', matches, allow_pickle=True)
    else:
        print("Loading matches...")
        matches = np.load('matches.npy', allow_pickle=True)

    sfm = SFM(features, matches, camera_model, args.reprojection_threshold)
    sfm.initialize()
    assert(sfm.check_observations_consisntency())
    if args.vis_frequency:
        sfm.visualize()
    iteration = 0
    while not sfm.is_finished():
        sfm.add_next_view()
        assert(sfm.check_observations_consisntency())
        iteration += 1
        if iteration % args.ba_frequency == 0:
            sfm.bundle_adjustment()
        if args.vis_frequency > 0 and iteration % args.vis_frequency == 0:
            sfm.visualize()
    print ("\n\nFINISHED\n\n")

    #save pointcloud
    with open('pointcloud.xyz', 'w') as f:
        for point in sfm.points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")
    sfm.visualize()

