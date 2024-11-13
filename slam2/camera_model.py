import numpy as np
#simple radial camera model from colmap
class RadialCameraModel:
    def __init__(self, size, focal_length, principal_point, distortion_coefficients):
        # Инициализация модели камеры с заданными параметрами
        self.size = size # Размер изображения в пикселях (ширина, высота)
        self.focal_length = focal_length # Фокусное расстояние камеры
        self.principal_point = principal_point # Координаты главной точки (оптического центра)
        self.distortion_coefficients = distortion_coefficients # Коэффициенты дисторсии (искажения)
    def resize(self, factor):
        # Метод изменения масштаба камеры с заданным коэффициентом factor
        self.size = (self.size * factor).astype(int) # Изменение размера изображения
        self.focal_length *= factor # Масштабирование фокусного расстояния
        self.principal_point *= factor # Масштабирование координат главной точки
        
    def project(self, points3d):
        valid = points3d[:, 2] > 0 # Проверяем, находятся ли точки перед камерой (z > 0)
        uv = np.zeros_like(points3d[:, :2]) # Создаем массив для хранения 2D координат проекций
        uv[valid] = points3d[:, :2][valid] / points3d[valid, 2][:, None] # Проецируем только точки, находящиеся перед камерой

        # Вычисляем радиальные координаты u^2 и v^2
        u2 = uv[:, 0] * uv[:, 0]
        v2 = uv[:, 1] * uv[:, 1]
        # Суммируем их для получения радиального расстояния r^2
        r2 = u2 + v2
        # Применяем коэффициенты дисторсии для корректировки искажений
        radial = 1 + self.distortion_coefficients * r2
        # Умножаем координаты на коэффициент дисторсии
        uv *= radial[:, None]
        # Преобразуем координаты из нормализованной системы в пиксели
        return uv * self.focal_length + self.principal_point


    def unproject(self, points2d):
        # Сдвигаем 2D координаты на главную точку
        points2d = points2d - self.principal_point
        # Делим на фокусное расстояние, чтобы перейти в нормализованное пространство
        points2d /= self.focal_length
        # Вычисляем радиальное расстояние r^2 для каждого 2D вектора
        r2 = np.sum(points2d ** 2, axis=1)
        # Вычисляем коэффициент для устранения радиального искажения
        factor = 1.0 / (1 + self.distortion_coefficients * r2)
        # Масштабируем 2D координаты, убирая искажения
        unprojected = points2d * factor[:, None] 

        # Добавляем координату z=1, чтобы получить 3D вектор в однородных координатах
        hom = np.hstack((unprojected, np.ones((unprojected.shape[0], 1))))
        return unprojected

    # Метод для вывода параметров модели камеры в строковом формате
    def __str__(self):
        return 'RadialCameraModel(size={}, focal_length={}, principal_point={}, distortion_coefficients={})'.format(
            self.size, self.focal_length, self.principal_point, self.distortion_coefficients)


class OpenCVCameraModel:
    def __init__(self, size, focal_length, principal_point, distortion_coefficients):
        # Initialize the camera model with specified parameters
        self.size = size  # Image size in pixels (width, height)
        self.focal_length = focal_length  # Focal lengths [fx, fy]
        self.principal_point = principal_point  # Principal point [cx, cy]
        self.distortion_coefficients = distortion_coefficients  # Distortion coefficients [k1, k2, p1, p2, k3]

    def resize(self, factor):
        # Scale the camera parameters by a specified factor
        self.size = (self.size * factor).astype(int)
        self.focal_length *= factor
        self.principal_point *= factor

    def project(self, points3d):
        # Check if points are in front of the camera (z > 0)
        valid = points3d[:, 2] > 0
        uv = np.zeros_like(points3d[:, :2])
        uv[valid] = points3d[:, :2][valid] / points3d[valid, 2][:, None]  # Normalize points

        # Apply distortion using OpenCV model
        r2 = np.sum(uv ** 2, axis=1)
        radial = 1 + self.distortion_coefficients[0] * r2 + \
                 self.distortion_coefficients[1] * r2 ** 2 + \
                 self.distortion_coefficients[4] * r2 ** 3
        uv[:, 0] += 2 * self.distortion_coefficients[2] * uv[:, 0] * uv[:, 1] + self.distortion_coefficients[3] * (
                    r2 + 2 * uv[:, 0] ** 2)
        uv[:, 1] += 2 * self.distortion_coefficients[3] * uv[:, 0] * uv[:, 1] + self.distortion_coefficients[2] * (
                    r2 + 2 * uv[:, 1] ** 2)
        uv *= radial[:, None]

        # Convert to pixel coordinates
        return uv * self.focal_length + self.principal_point

    def unproject(self, points2d):
        # Shift points by principal point
        points2d = points2d - self.principal_point
        # Normalize to the focal plane
        points2d /= self.focal_length
        # Estimate radial distortion to remove it (approximate iterative method may be needed for complex cases)
        r2 = np.sum(points2d ** 2, axis=1)
        radial = 1 / (1 + self.distortion_coefficients[0] * r2 + self.distortion_coefficients[1] * r2 ** 2 +
                      self.distortion_coefficients[4] * r2 ** 3)
        undistorted = points2d * radial[:, None]

        # Homogeneous coordinates for 3D
        hom = np.hstack((undistorted, np.ones((undistorted.shape[0], 1))))
        return hom

    def __str__(self):
        return 'OpenCVCameraModel(size={}, focal_length={}, principal_point={}, distortion_coefficients={})'.format(
            self.size, self.focal_length, self.principal_point, self.distortion_coefficients)

def parse_camera_model(text):
    # Функция для разбора текстового представления модели камеры и создания объекта RadialCameraModel
    # Разделяем строку на компоненты
    parts = text.split()
    # Первый элемент - название модели камеры
    model = parts[0]
    # Проверяем, является ли модель SIMPLE_RADIAL
    if model == 'SIMPLE_RADIAL':
        # Извлекаем размер изображения
        size = np.array([int(parts[1]), int(parts[2])])
        # Извлекаем фокусное расстояние
        focal_length = float(parts[3])
        # Извлекаем координаты главной точки
        principal_point = np.array([float(parts[4]), float(parts[5])])
        # Извлекаем коэффициент дисторсии
        distortion_coefficients = float(parts[6])
        # Создаем и возвращаем объект RadialCameraModel с заданными параметрами
        return RadialCameraModel(size, focal_length, principal_point, distortion_coefficients)

    elif model == 'OPENCV':
        size = np.array([int(parts[1]), int(parts[2])])
        focal_length = np.array([float(parts[3]), float(parts[4])])
        principal_point = np.array([float(parts[5]), float(parts[6])])
        distortion_coefficients = np.array([float(parts[7]), float(parts[8]), float(parts[9]), float(parts[10]), float(parts[11])])
        return OpenCVCameraModel(size, focal_length, principal_point, distortion_coefficients)

    else:
        # Если модель неизвестна, вызываем ошибку
        raise ValueError('Unknown camera model: {}'.format(model))
