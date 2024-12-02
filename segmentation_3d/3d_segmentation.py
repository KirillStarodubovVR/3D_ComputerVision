import pandas as pd
import numpy as np
import scipy
import open3d as o3d
from tqdm import tqdm
import os
from typing import List, Dict,Tuple,Any
from pathlib import Path
import catboost
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split, KFold, GroupKFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import plotly.graph_objects as go
from plotly.subplots import make_subplots

df_mapping = pd.read_csv("filtering_mapping.txt", header=None)
# Находим строку, содержащую текст 'removed from the data set'.
row = df_mapping[df_mapping[0].str.contains('removed from the data set')].index

# Отбираем строки до найденного индекса, исключая строки после 'removed from the data set'.
# https://stackoverflow.com/questions/31593201/how-are-iloc-and-loc-different
df_mapping = df_mapping.iloc[:row[0]-1]
# df_mapping = df_mapping.iloc[:row[0]]

# Удаляем строки, содержащие заголовок 'Remapped labeled:'.
df_mapping = df_mapping[~df_mapping[0].str.contains('Remapped labeled:')]
# Разбиваем строки в колонке 0 на две части по разделителю ' -- '.
df_mapping[['label', 'name']] = df_mapping[0].str.split(' -- ', expand=True)
# Удаляем исходную колонку 0.
df_mapping = df_mapping.drop(0, axis=1)
# Удаляем лишние пробелы в значениях колонки label.
df_mapping["label"] = df_mapping["label"].str.strip()

# df_mapping.head()
# Обновляем индексы DataFrame
# df_mapping.index = range(len(df_mapping))
df_mapping.index = pd.RangeIndex(0, len(df_mapping), 1)


# Нам нужны только 5 классов : scatter_misc, default_wire, utility_pole, load_bearing, facade
label_map = {0:'1004',1:'1100',2:'1103',3:'1200',4:'1400'}


# Задача

class PointCloudDataset(object):
    def __init__(self,data_path: str,grouping_method: str,neighbourhood_th: Any,label_map: Dict):
          """
          In :
              data_path: str - путь до папки с данными
              grouping_method : str - метод поиска соседей , ["knn","radius_search",имплементированный вами]
              neighbourhood_th : Any[int,float] - пороговое значение для k - количества соседей или radius - радиуса сферы
              label_map : Dict - словарь {label : index}
          """

          self.data_path = data_path
          self.grouping_method = grouping_method
          self.neighbourhood_th = neighbourhood_th

          self.label_map = label_map
          self.inv_label_map = {int(v): k for k, v in self.label_map.items()}

          self.feature_names = ['x', 'y', 'z', 'eigenvals_sum', 'linearity', 'planarity', 'change_of_curvature',
                                 'scattering', 'omnivariance', 'anisotropy', 'eigenentropy', 'label','scene_id']

    def read_points_from_file(self, filename: str) -> Tuple[np.ndarray,np.ndarray]:
        """
        In :
            filename: str  Путь до файла с облаком точек
        Out :
            points,labels : Tuple[np.ndarray,np.ndarray] -> массивы точек , лейблов
        """

        data = np.loadtxt(filename)
        points = data[:, 0:3]  # (x, y, z)
        labels = data[:, 3].astype(np.int32)    # label
        return points, labels

    def load_from_directory(self, directory: str) -> Tuple[List[np.ndarray],List[np.ndarray]]:
        """
        In :
            directory: str  Путь до директории с файлами
        Out :
            all_points, all_labels : Tuple[List[np.ndarray],List[np.ndarray]] Набор точек,лейблов для каждой сцены
        """

        all_points = []
        all_labels = []
        for filename in os.listdir(directory):
            if filename.startswith("oakland_part") and filename.endswith(".xyz_label_conf"):
                file_path = os.path.join(directory, filename)
                points, labels = self.read_points_from_file(file_path)
                all_points.append(points)
                all_labels.append(labels)

        return all_points, all_labels

    def create_kdtree(self, points :np.ndarray)-> Tuple[o3d.geometry.PointCloud,o3d.geometry.KDTreeFlann]:
        """
        In :
            points: np.ndarray  Облако точек
        Out :
            pcd, tree : Tuple[o3d.geometry.PointCloud,o3d.geometry.KDTreeFlann] - облако точек, k-d дерево
        """
        # https://www.open3d.org/docs/latest/tutorial/geometry/kdtree.html
        # Создание облака точек
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        tree = o3d.geometry.KDTreeFlann(pcd)

        return pcd, tree

    def knn(self, pcd: o3d.geometry.PointCloud, tree: o3d.geometry.KDTreeFlann, query_index : int, k: int) -> np.ndarray:
        """
        In :
            pcd: o3d.geometry.PointCloud  Облако точек
            tree : o3d.geometry.KDTreeFlann k - d дерево (https://www.open3d.org/docs/latest/python_api/open3d.geometry.KDTreeFlann.html)
            query_index : int Индекс точки в дереве для которой нужно найти соседей
            k : int Количество ближайших соседей для поиска
        Out :
            points: np.ndarray  -> найденные точки(включая query_index)
        """
        # https://www.open3d.org/docs/latest/python_api/open3d.geometry.KDTreeFlann.html
        [k, indices, _] = tree.search_knn_vector_3d(pcd.points[query_index], k)

        return np.array(pcd.points)[indices]

    def radius_search(self, pcd: o3d.geometry.PointCloud, tree: o3d.geometry.KDTreeFlann, query_index : int, radius: float) -> np.ndarray:
        """
        In :
            pcd: o3d.geometry.PointCloud  Облако точек
            tree : o3d.geometry.KDTreeFlann k - d дерево (https://www.open3d.org/docs/latest/python_api/open3d.geometry.KDTreeFlann.html)
            query_index : int Индекс точки в дереве для которой нужно найти соседей
            radius : float Радиус сферы , в метрах
        Out :
            points: np.ndarray  -> найденные точки(включая query_index)
        """
        # https://www.open3d.org/docs/latest/python_api/open3d.geometry.KDTreeFlann.html
        [k, indices, _] = tree.search_radius_vector_3d(pcd.points[query_index], radius)

        return np.array(pcd.points)[indices]

    def get_eugen_stats(self, neighbourhood_points: np.ndarray) -> Tuple[float, ...]:
        """
        In :
            neighbourhood_points: np.ndarray  Облако соседних точек найденных с помощью knn или radius_search
        Out :
            features: Tuple[float, ...]  -> признаки вычисленные по данному облаку точек
        """
        # https://mediatum.ub.tum.de/doc/800632/941254.pdf
        # Шаг 1. Найдем собственные значения:
        # o3d ищет по дефолту смещённую оценку n
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(neighbourhood_points)
        # o3d_mean_cov = pcd.compute_mean_and_covariance()

        # Шаг 2. Найдем собственные значения:
        # Найдём центр масс neighbourhood_points
        centroid = np.mean(neighbourhood_points, axis=0)
        # Смещаем точки на величину среднего значения
        centred_pts = neighbourhood_points - centroid
        # Находим ковариационную матрицу (по дефолту не смещённая)
        cov_matrix = np.cov(centred_pts, rowvar=False, ddof=1)
        # Проверка расчёта матрицы вручную
        # M = np.dot(centred_pts.T, centred_pts)/(centred_pts.shape[0]-1) - не смещённая
        # M = np.dot(centred_pts.T, centred_pts)/(centred_pts.shape[0]) - смещённая

        eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
        # Clip в случае отрицательных значений
        eigen_values = np.clip(eigen_values, a_min=0, a_max=None)
        # Сортировка по убыванию e1>e2>e3
        eigen_values = np.sort(eigen_values)[::-1]
        e1, e2, e3 = eigen_values
        # Шаг 3. Вычислим признаки:
        # sum of eigenvalues
        sum_of_eigenvalues = np.sum(eigen_values)
        # linearity
        linearity = (e1-e2)/(e1 + 1e-6)
        # planarity
        planarity = (e2-e3)/(e1 + 1e-6)
        # scattering
        scattering = e3/(e1 + 1e-6)
        # omnivariance
        omnivariance = np.cbrt(np.prod(eigen_values))
        # anisotropy
        anisotropy = (e1-e3)/(e1 + 1e-6)
        # eigentropy
        eigentropy = -np.sum(eigen_values/(sum_of_eigenvalues + 1e-6) * np.log(eigen_values/(sum_of_eigenvalues + 1e-6)))
        # change of curvative
        change_of_curvature = e3 / (sum_of_eigenvalues + 1e-6)

        return sum_of_eigenvalues, linearity, planarity, change_of_curvature, \
        scattering, omnivariance, anisotropy, eigentropy

    def get_eugen_stats_mod(self, neighbourhood_points: np.ndarray) -> Tuple[float, ...]:
        """
        In :
            neighbourhood_points: np.ndarray  Облако соседних точек найденных с помощью knn или radius_search
        Out :
            features: Tuple[float, ...]  -> признаки вычисленные по данному облаку точек
        """
        # https://mediatum.ub.tum.de/doc/800632/941254.pdf
        # Шаг 1. Найдем собственные значения:
        # o3d ищет по дефолту смещённую оценку n
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(neighbourhood_points)
        # o3d_mean_cov = pcd.compute_mean_and_covariance()

        # Шаг 2. Найдем собственные значения:
        # Найдём центр масс neighbourhood_points
        centroid = np.mean(neighbourhood_points, axis=0)
        # Смещаем точки на величину среднего значения
        centred_pts = neighbourhood_points - centroid
        # Находим ковариационную матрицу (по дефолту не смещённая)
        cov_matrix = np.cov(centred_pts, rowvar=False, ddof=1)
        # Проверка расчёта матрицы вручную
        # M = np.dot(centred_pts.T, centred_pts)/(centred_pts.shape[0]-1) - не смещённая
        # M = np.dot(centred_pts.T, centred_pts)/(centred_pts.shape[0]) - смещённая

        eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
        # Clip в случае отрицательных значений
        eigen_values = np.clip(eigen_values, a_min=0, a_max=None)
        # Сортировка по убыванию e1>e2>e3
        eigen_values = np.sort(eigen_values)[::-1]
        e1, e2, e3 = eigen_values
        # Шаг 3. Вычислим признаки:
        # sum of eigenvalues
        sum_of_eigenvalues = np.sum(eigen_values)
        # linearity
        linearity = (e1-e2)/(sum_of_eigenvalues + 1e-6)
        # planarity
        planarity = (e2-e3)/(sum_of_eigenvalues + 1e-6)
        # scattering
        scattering = e3/(sum_of_eigenvalues + 1e-6)
        # omnivariance
        omnivariance = np.cbrt(np.prod(eigen_values))
        # anisotropy
        anisotropy = (e1-e3)/(sum_of_eigenvalues + 1e-6)
        # eigentropy
        normalized_eigen_values = eigen_values/(sum_of_eigenvalues + 1e-6)
        normalized_eigen_values = np.clip(normalized_eigen_values, 1e-6, 1)
        # eigentropy = -np.sum(eigen_values/(sum_of_eigenvalues + 1e-6) * np.log(eigen_values/(sum_of_eigenvalues + 1e-6)))
        eigentropy = -np.sum(normalized_eigen_values * np.log(normalized_eigen_values))
        # change of curvative
        change_of_curvature = e3 / (sum_of_eigenvalues + 1e-6)

        return sum_of_eigenvalues, linearity, planarity, change_of_curvature, \
        scattering, omnivariance, anisotropy, eigentropy

    def create_dataset(self) -> pd.DataFrame:
        """
        In :

        Out :
            dataframe : pd.DataFrame Датафрейм с данными, согласно названиям колонок из self.feature_names
        """
        dataset = []
        # Шаг 1. Загрузка данных всех сцен из указанной директории(self.data_path) :
        all_points, all_labels = self.load_from_directory(self.data_path)

        # Шаг 2. Итерирование по сценам:
        for scene_id, (pts, labels) in tqdm(enumerate(zip(all_points, all_labels)), desc="Scene"):
            # Шаг 3. Создание kdtree:
            pcd, tree = self.create_kdtree(pts)
            # Шаг 4. Итерирование по всем точкам из kdtree:
            for i in range(len(pts)):
                # Шаг 5. Поиск соседей одним из методов - knn или radius search:
                if self.grouping_method == "knn":
                    nb_pts = self.knn(pcd, tree, i , self.neighbourhood_th)
                    # Шаг 6. Вычисление признаков:
                    features = self.get_eugen_stats_mod(nb_pts)

                elif self.grouping_method == "radius_search":
                    nb_pts = self.radius_search(pcd, tree, i , self.neighbourhood_th)
                    # Шаг 6. Вычисление признаков:
                    features = self.get_eugen_stats_mod(nb_pts)

                elif self.grouping_method == "msn":
                    features = ()

                    for radius in self.neighbourhood_th:
                        nb_pts = self.radius_search(pcd, tree, i, radius)
                        if len(nb_pts)>3:
                            feature = self.get_eugen_stats_mod(nb_pts)
                        else:
                            feature = (0,) * 8

                        features = features + feature

                # Шаг 7. Заполнение списка описанием точки [x,y,z,features,label,scene_id] - 1x13:
                dataset.append(list(pts[i]) + list(features) + [self.inv_label_map[labels[i]], scene_id])

        if self.grouping_method == "msn":
            columns = []
            for radius in self.neighbourhood_th:
                columns.extend([f"eigen_sum_{radius}", f"linearity_{radius}", f"planarity_{radius}",
                                f"change_of_curvature_{radius}", f"scattering_{radius}", f"omnivariance_{radius}",
                                f"anisotropy_{radius}", f"eigenentropy_{radius}"])

            self.feature_names = self.feature_names[:3] + columns + self.feature_names[-2:]

        #Шаг 8. Формирование DataFrame:
        dataframe = pd.DataFrame(dataset, columns=self.feature_names)

        return dataframe


train_dataframe_path = Path("train_dataframe.pkl")

if train_dataframe_path.is_file():
    print("The file 'train_dataframe_path' exists. Start loading.....")
    train_dataframe = pd.read_pickle(train_dataframe_path)
else:
    print("The file 'train_dataframe_path' does not exist.")
    train_data_path = "./training" # Путь до тренировочных данных
    method = "msn" # Метод поиска соседей | knn
    neighbourhood_th = [6, 9, 12] # Количество соседей или радиус | 10
    train_dataset = PointCloudDataset(train_data_path,method,neighbourhood_th,label_map)
    train_dataframe = train_dataset.create_dataset()
    train_dataframe.to_pickle("train_dataframe.pkl")

test_dataframe_path = Path("test_dataframe.pkl")

if test_dataframe_path.is_file():
    print("The file 'test_dataframe_path' exists. Start loading.....")
    test_dataframe = pd.read_pickle(test_dataframe_path)
else:
    print("The file 'train_dataframe_path' does not exist.")
    test_data_path = "./testing"
    method = "msn" # Метод поиска соседей | knn
    neighbourhood_th = [6, 9, 12] # Количество соседей или радиус | 10
    test_dataset = PointCloudDataset(test_data_path,method,neighbourhood_th,label_map)
    test_dataframe = test_dataset.create_dataset()
    test_dataframe.to_pickle("test_dataframe.pkl")


# Карта цветов для классов
color_map = {
    0: (0, 1, 0),      # scatter_misc
    1: (1, 1, 0),      # default_wire
    2: (0, 0, 1),      # utility_pole
    3: (0.4, 0, 0.6),  # load_bearing
    4: (0.5, 0.5, 0.5) # facade
}


def visualize_point_cloud(points: np.ndarray, labels: np.ndarray) -> go.Figure:
    """
    Визуализация облака точек с цветами, заданными индексами классов.

    In :
       points : np.ndarray - Облако точек (x, y, z) [N, 3]
       labels : np.ndarray - Индексы классов для каждой точки [N]
    Out :
       fig : go.Figure - 3D график облака точек
    """
    # Создаем список цветов для каждой точки на основе color_map
    colors = np.array([color_map[label] for label in labels])

    # Создаем 3D график облака точек
    scatter = go.Scatter3d(
        x=points[:, 0],  # Координаты x
        y=points[:, 1],  # Координаты y
        z=points[:, 2],  # Координаты z
        mode='markers',  # Используем маркеры для отображения точек
        marker=dict(
            size=3,  # Размер маркеров
            color=colors,  # Цвета маркеров
            opacity=0.8  # Прозрачность маркеров
        )
    )

    # Настраиваем фигуру
    fig = go.Figure(data=[scatter])
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),
        title='Point Cloud Visualization',
    )

    return fig

points  = test_dataframe.to_numpy()[:,:3]
labels = test_dataframe.to_numpy()[:,-2]
# colors = [color_map[label] for label in labels]

fig = visualize_point_cloud(points, labels)
fig.show()

scene_0 = train_dataframe[train_dataframe["scene_id"] == 0]
points  = scene_0.to_numpy()[:,:3]
labels = scene_0.to_numpy()[:,-2]
# colors = [color_map[label] for label in labels]

fig = visualize_point_cloud(points, labels)
fig.show()

class Trainer(object):

    def __init__(self,train_data : pd.DataFrame,
                 test_data: pd.DataFrame,
                 model_config: Dict,
                 n_folds: int,
                 use_class_weights: bool,
                 use_oversample: bool):
        """
        In :
          train_data: pd.DataFrame - таблица с данными для тренировки
          test_data: pd.DataFrame - таблица с данными для тестирования
          model_config : Dict - словарь с параметрами модели
          n_folds : int - количество фолдов
        """

        self.model_params = model_config
        self.n_folds = n_folds
        self.train_data = train_data
        self.test_data = test_data
        self.metrics = ['TotalF1:average=Weighted;use_weights=False', 'Accuracy']
        self.folds = self.create_folds()
        self.use_class_weights = use_class_weights
        self.use_oversample = use_oversample

        self.class_weights = None



    def create_folds(self) -> List[List[np.ndarray]]:
        """
        In :
          self.train_data : pd.DataFrame  - тренировочные данные
          self.n_folds : int - количество фолдов
        Out :
          kfold_dset_index : List[List[np.ndarray]] - список пар индексов train_index,val_index для каждого из n фолдов
        """
        # StratifiedKFold
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
        # This cross-validation object is a variation of KFold that returns stratified folds.
        # The folds are made by preserving the percentage of samples for each class.
        # To provide stratified fold we need

        kfold_dset_index = []
        y = self.train_data["label"]
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for train_index, val_index in skf.split(self.train_data, y):
            kfold_dset_index.append((np.array(train_index), np.array(val_index)))


        return kfold_dset_index

    def train_catboost(self, train_pool: catboost.Pool, val_pool: catboost.Pool) -> catboost.CatBoostClassifier:
        """
        In :
          train_pool : catboost.Pool - инициализированный конструктора для тренировочных данных (относящихся к train_index)
          val_pool : catboost.Pool - инициализированный конструктора для валидационных данных (относящихся к val_index)
        Out :
          catboost_model : catboost.CatBoostClassifier - обученная модель
        """
        params = self.model_params.copy()
        if self.use_class_weights:
            params['class_weights'] = self.class_weights

        # Инициализация и обучение модели
        catboost_model = CatBoostClassifier(**params)

        catboost_model.fit(train_pool, eval_set=val_pool, verbose=False)

        return catboost_model

    def test_catboost(self, test_pool: catboost.Pool, catboost_model:catboost.CatBoostClassifier) -> List[float]:
        """
        In :
          test_pool : catboost.Pool - инициализированный коеструктор для тестовых данных (относящихся к self.test_data)
          catboost_model : catboost.CatBoostClassifier - обученная модель
        Out :
          metric_values : List[float] - значение метрик для комбинации всех деревьев (https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier_eval-metrics ntree_end)
        """

        # Подсчет метрик на тестовой выборке с использованием обученной модели
        metrics = catboost_model.eval_metrics(test_pool, metrics=self.metrics, ntree_end=catboost_model.best_iteration_)
        # первая метрика f1 вторая accuracy
        metric_values = [np.mean(metrics[metric]) for metric in metrics]

        return metric_values

    def compute_class_weights(self, y:np.ndarray):
        """
        Вычислить веса классов.
        Args:
            y (np.ndarray): Массив меток классов.
        Returns:
            dict: Словарь весов классов.
        """
        classes = np.unique(y)
        class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        # self.class_weights = dict(zip(classes, class_weights))
        self.class_weights = class_weights

    def fit(self):
        """
        In :
          self.train_data: pd.DataFrame - таблица с данными для тренировки
          self.test_data: pd.DataFrame - таблица с данными для тестирования
          self.folds : List[List[np.ndarray]] - список пар индексов train_index,val_index для каждого из n фолдов

        Out :
          self.f1_list : List[float] - список из значений метрики F1(average = Weighted) на тестовой выборке для каждой из обученных моделей
          self.acc_list : List[float] - список из значений метрики Accuracy на тестовой выборке для каждой из обученных моделей
          self.models_list : List[catboost.CatBoostClassifier] - список из обученных на каждом из фолдов моделей
        """

        # Шаг 1. Сформируем X,y X_test y_test убрав нужные колонки из pd.DataFrame :
        X = self.train_data.drop(columns=["label","scene_id"])
        y = self.train_data['label']

        X_test = self.test_data.drop(columns=["label","scene_id"])
        y_test = self.test_data['label']

        # Шаг 2. Проинициализируем self.f1_list,self.acc_list,self.models_list :
        self.f1_list=[]
        self.acc_list=[]
        self.models_list = []

        # Шан 3. Итерирование по self.folds :
        for train_idx, val_idx in self.folds:

            # Шаг 4. Создадим X_train,y_train X_val,y_val используя индексы фолдов:
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]  # Тренировочные данные
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]  # Валидационные данные

            # Шаг 5. Инициализируем train_pool , val_pool:
            train_pool = Pool(X_train, y_train)
            val_pool = Pool(X_val, y_val)

            # Шаг 5. self.train_catboost:
            # Уменьшаем влияние дисбаланса классов
            if self.use_class_weights:
                self.compute_class_weights(y_train)

            catboost_model = self.train_catboost(train_pool, val_pool)

            # Шаг 6. Инициализируем test_pool из X_test,y_test :
            test_pool = Pool(X_test, y_test)

            # Шаг 7. self.test_catboost:
            metric_values = self.test_catboost(test_pool, catboost_model)

            # Шаг 8. Запишем полученные метрики и модель в созданные списки:
            f1, acc = metric_values
            self.f1_list.append(f1)
            self.acc_list.append(acc)
            self.models_list.append(catboost_model)

            del X_train,y_train,X_val,y_val,train_pool,val_pool,test_pool,catboost_model

        # Шаг 9. Выведем среднее и стандартное отклонение полученных по n фолдам метрик
        print('\n================================')
        print(f"F1={np.mean(self.f1_list)} +/- {np.std(self.f1_list)}")
        print(f"Accuracy={np.mean(self.acc_list)} +/- {np.std(self.acc_list)}")

    def ensemble_prediction(self, X_test):
        all_labels = []
        for model in self.models_list:
            labels = model.predict(X_test)
            all_labels.append(labels)

        stacked_vectors = np.column_stack(all_labels)

        final_labels, _ = scipy.stats.mode(stacked_vectors, axis=1)

        return final_labels.flatten().reshape(-1, 1)

    def return_best(self):
        """
        In :
          self.f1_list : List[float] - список из значений метрики F1(average = Weighted) на тестовой выборке для каждой из обученных моделей
          self.models_list : List[catboost.CatBoostClassifier] - список из обученных на каждом из фолдов моделей

        Out :
          self.f1_list : List[float] - список из значений метрики F1(average = Weighted) на тестовой выборке для каждой из обученных моделей
          self.acc_list : List[float] - список из значений метрики Accuracy на тестовой выборке для каждой из обученных моделей
          self.models_list : List[catboost.CatBoostClassifier] - список из обученных на каждом из фолдов моделей
        """
        # Выбор лучшей из n обученных моделей на базе значений метрики F1
        best_idx = np.argmax(self.f1_list)
        best_model = self.models_list[best_idx]

        return best_model

# Обучаем модель на n-fold
params = {"iterations": 100,
          "depth": 3,
          "loss_function": "MultiClass",
          "verbose": False}
n_folds = 3 # количество фолдов
use_class_weights = False
use_oversample = False
trainer = Trainer(train_dataframe, test_dataframe, params, n_folds, use_class_weights, use_oversample)
trainer.fit()

# Получаем наилучшую модель
best_model = trainer.return_best()

# Получим предсказание для тестового набора данных
X_test = test_dataframe.drop(columns=["label","scene_id"])
y_test = test_dataframe['label']
y_pred = best_model.predict(X_test)

# Визуализация тестовых данных с оригинальными лейблами
points  = test_dataframe.to_numpy()[:,:3]
labels = test_dataframe.to_numpy()[:,-2]
# colors = [color_map[label] for label in labels]
visualize_point_cloud(points, labels)
fig = visualize_point_cloud(points, labels)
fig.show()

# Визуализация тестовых данных с предсказанными лейблами
points  = test_dataframe.to_numpy()[:,:3]
labels = y_pred.reshape(-1,)
# colors = [color_map[label] for label in labels]
visualize_point_cloud(points, labels)
fig = visualize_point_cloud(points, labels)
fig.show()

# Использование ансамбля моделей для предсказания
# y_pred = trainer.ensemble_prediction(X_test)
# # Визуализация тестовых данных с предсказанными лейблами
# points  = test_dataframe.to_numpy()[:,:3]
# labels = y_pred.reshape(-1,)
# # colors = [color_map[label] for label in labels]
# visualize_point_cloud(points, labels)
# fig = visualize_point_cloud(points, labels)
# fig.show()



print("Finish")