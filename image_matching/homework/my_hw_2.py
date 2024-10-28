from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import logging
import skimage.io as io
from transformers import AutoImageProcessor, SuperPointForKeypointDetection
import torch
from PIL import Image
import requests


# Читаем данные
# images_paths = list(Path('./hw_imgs').glob('./photo_*'))
images_paths = list(Path('./imgs').glob('./image_*'))
images_paths = sorted(images_paths)
image_list = []
for image_path in images_paths:
    image = cv2.imread(image_path.as_posix())
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_list.append(image)

num_images = len(image_list)

# f, axarr = plt.subplots(1,num_images, figsize=(15,10))
# for i in range(len(image_list)) :
#     axarr[i].imshow(image_list[i])
#     axarr[i].axis("off")
# plt.show()

processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")

def evaluate_descriptor_for_image_superpoint(image : np.array, scale_factor: int = 1):


    # Извлекаем исходную высоту, ширину изображения и количество цветовых каналов
    orig_h, orig_w, _ = image.shape

    # Если задан масштабный коэффициент, изменяем размер изображения
    if scale_factor != 1 :
        image = cv2.resize(image, (orig_w // scale_factor, orig_h // scale_factor))

    inputs = processor(image, return_tensors="pt")

    outputs = model(**inputs)

    # Обнаруживаем ключевые точки и вычисляем дескрипторы с помощью SuperPoint
    keypoints, descriptors = outputs.keypoints, outputs.descriptors

    # Сохраняем исходные ключевые точки для последующего рисования
    keypoints_orig = keypoints

    # Преобразуем ключевые точки в массив numpy для удобства дальнейших вычислений
    # keypoints = np.asarray([[*kp.pt] for kp in keypoints], dtype=np.int32)
    # keypoint_x, keypoint_y = int(keypoint[0].item()), int(keypoint[1].item())

    # Масштабируем ключевые точки обратно в исходные размеры изображения
    keypoints[:, 0] = (keypoints[:, 0] / (orig_w // scale_factor)) * orig_w
    keypoints[:, 1] = (keypoints[:, 1] / (orig_h // scale_factor)) * orig_h

    # Возвращаем исходные ключевые точки, масштабированные ключевые точки и дескрипторы
    return keypoints_orig.numpy()[0], keypoints.numpy()[0], descriptors.detach().numpy()[0]


def match_keypoints(kpsA,
                    kpsB,
                    featuresA,
                    featuresB,
                    ratio=0.75,
                    reprojThresh=4.0):

    # Создаем объект FlannBasedMatcher для поиска ближайших соседей между дескрипторами двух изображений
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

    # Выполняем сопоставление с использованием k-NN (2 ближайших соседа для каждого дескриптора)
    matches = flann.knnMatch(featuresA, featuresB, k=2)

    # Инициализируем список для хранения "хороших" совпадений
    good = []

    # Проходим по каждому набору пар ближайших соседей (m - лучший матч, n - второй лучший)
    # По факту i тут не нужен
    for i, (m, n) in enumerate(matches):
        # Если расстояние до ближайшего соседа значительно меньше, чем до второго (по критерию Лоу),
        # то добавляем этот матч в список хороших
        if m.distance < 0.7 * n.distance:
            good.append(m)

    # Преобразуем "хорошие" матчи в массив индексов (индексы соответствующих ключевых точек на изображениях A и B)
    matches = np.asarray([[m.trainIdx, m.queryIdx] for m in good], dtype=np.int32)

    # Если найдено хотя бы 4 совпадения, можно вычислить гомографию
    if len(matches) > 4:
        # Извлекаем координаты точек на двух изображениях, соответствующих матчам
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])

        # Вычисляем матрицу гомографии с использованием алгоритма RANSAC
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

        # Возвращаем матчи, матрицу гомографии и статус каждой точки (успешное или неуспешное совпадение)
        return (matches, H, status)

    # Если недостаточно совпадений для вычисления гомографии, возвращаем None
    return None


def draw_matches(imageA,
                 imageB,
                 kpsA,
                 kpsB,
                 matches,
                 status):
    # Извлекаем размеры двух изображений
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]

    # Создаем пустое изображение для визуализации, которое объединяет два изображения по горизонтали
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")

    # Копируем изображение A в левую часть визуализации
    vis[0:hA, 0:wA] = imageA

    # Копируем изображение B в правую часть визуализации
    vis[0:hB, wA:] = imageB

    # Проходим по каждому совпадению и статусу (если статус равен 1, совпадение считается успешным)
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        # Обрабатываем совпадение, если оно было успешно
        if s == 1:
            # Извлекаем координаты ключевых точек на изображении A
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))

            # Извлекаем координаты ключевых точек на изображении B, сдвинутые на ширину изображения A
            ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))

            # Рисуем линию, соединяющую соответствующие точки на двух изображениях
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

    # Возвращаем изображение с визуализированными совпадениями
    return vis


def blending_mask(height,
                  width,
                  barrier,
                  smoothing_window,
                  left_biased=True):
    # Утверждаем, что барьер меньше ширины изображения
    assert barrier < width

    # Создаем пустую маску (матрица нулей) с размерами изображения (высота, ширина)
    mask = np.zeros((height, width))

    # Определяем смещение как половину окна сглаживания
    offset = int(smoothing_window / 2)

    try:
        # Если маска должна быть смещена влево
        if left_biased:
            # Вставляем линейный градиент от 1 до 0 в области вокруг барьера (на границе смешивания)
            mask[:, barrier - offset: barrier + offset + 1] = np.tile(
                np.linspace(1, 0, 2 * offset + 1).T, (height, 1))

            # Область слева от барьера полностью заполняем единицами (маска полностью белая)
            mask[:, : barrier - offset] = 1

        else:
            # Вставляем линейный градиент от 0 до 1 в области вокруг барьера (для правого смещения)
            mask[:, barrier - offset: barrier + offset + 1] = np.tile(
                np.linspace(0, 1, 2 * offset + 1).T, (height, 1))

            # Область справа от барьера заполняем единицами (маска полностью белая)
            mask[:, barrier + offset:] = 1

    except BaseException:
        # Если происходит исключение (например, неправильные размеры), применяем альтернативную логику
        if left_biased:
            mask[:, barrier - offset: barrier + offset + 1] = np.tile(
                np.linspace(1, 0, 2 * offset).T, (height, 1)
            )
            mask[:, : barrier - offset] = 1
        else:
            mask[:, barrier - offset: barrier + offset + 1] = np.tile(
                np.linspace(0, 1, 2 * offset).T, (height, 1)
            )
            mask[:, barrier + offset:] = 1

    # Возвращаем трехканальную маску (добавляем дополнительные каналы для цветных изображений)
    return cv2.merge([mask, mask, mask])


def panorama_blending(dst_img_rz,
                      src_img_warped,
                      width_dst,
                      side):
    # Функция для смешивания двух изображений
    # Извлекаем высоту, ширину и количество цветовых каналов у изображений
    h, w, _ = dst_img_rz.shape

    # Определяем размер окна сглаживания как 1/8 ширины целевого изображения
    smoothing_window = int(width_dst / 8)

    # Устанавливаем барьер для смешивания, немного смещенный от края изображения
    barrier = width_dst - int(smoothing_window / 2)

    # Вычисляем маску для изображения справа (или основной)
    mask1 = blending_mask(h,
                            w,
                            barrier,
                            smoothing_window=smoothing_window,
                            left_biased=True
                            )

    # Вычисляем маску для изображения слева (соседнего для стыковки)
    mask2 = blending_mask(h,
                            w,
                            barrier,
                            smoothing_window=smoothing_window,
                            left_biased=False
                            )

    if side == "left":
        dst_img_rz = cv2.flip(dst_img_rz, 1)
        src_img_warped = cv2.flip(src_img_warped, 1)

    # TODO: Смешивание двух изображений на основе масок mask1 и mask2
    pano = (dst_img_rz * mask1 + src_img_warped * mask2).astype(np.uint8)

    if side == "left":
        pano = cv2.flip(pano, 1)

    # fig, ax = plt.subplots(1,3, figsize=(15,10))
    # ax[0].imshow((src_img_warped * mask2).astype(np.uint8))
    # ax[1].imshow((dst_img_rz * mask1).astype(np.uint8))
    # ax[2].imshow(pano)

    return pano


def crop(panorama,
         h_dst,
         conners):
    # Находим минимальные и максимальные координаты углов по осям X и Y
    [x_min, y_min] = np.int32(conners.min(axis=0).ravel() - 0.5)

    # Создаем смещение для сдвига изображения в положительные координаты (если нужно)
    t = [-x_min, -y_min]

    # Преобразуем углы в целочисленный формат
    conners = conners.astype(int)

    # Если левый верхний угол изображения имеет отрицательную координату X (изображение смещено влево)
    if conners[0][0][0] < 0:

        # Находим ширину изображения, которая лежит в области слева от барьера
        n = abs(-conners[1][0][0] + conners[0][0][0])
        # Обрезаем панораму, удаляя лишние пиксели слева
        panorama = panorama[t[1]: h_dst + t[1], n:, :]
    else:

        # Если изображение полностью внутри области панорамы, обрезаем по правому краю
        if conners[2][0][0] < conners[3][0][0]:
            # Обрезаем панораму по X-координатам нижних углов
            panorama = panorama[t[1]: h_dst + t[1], 0: conners[2][0][0], :]
        else:
            panorama = panorama[t[1]: h_dst + t[1], 0: conners[3][0][0], :]
    # Возвращаем обрезанную панораму

    # plt.figure(figsize=(20,5))
    # plt.imshow(panorama.astype(np.uint8))

    return panorama


def warp_two_images_superpoint(src_img, dst_img):
    # Вычисляем ключевые точки и дескрипторы для исходного изображения (src_img)
    _, keypoints_src, descriptors_src = evaluate_descriptor_for_image_superpoint(src_img)
    # Вычисляем ключевые точки и дескрипторы для целевого изображения (dst_img)
    _, keypoints_dst, descriptors_dst = evaluate_descriptor_for_image_superpoint(dst_img)

    # Находим матрицу гомографии (H) и совпадающие ключевые точки между двумя изображениями
    (matches, H, _) = match_keypoints(keypoints_src,
                                      keypoints_dst,
                                      descriptors_src,
                                      descriptors_dst,
                                      True)

    # get height and width of two images
    height_src, width_src = src_img.shape[:2]
    height_dst, width_dst = dst_img.shape[:2]

    # extract conners of two images: top-left, bottom-left, bottom-right, top-right
    pts1 = np.float32(
        [[0, 0], [0, height_src], [width_src, height_src], [width_src, 0]]
    ).reshape(-1, 1, 2)
    pts2 = np.float32(
        [[0, 0], [0, height_dst], [width_dst, height_dst], [width_dst, 0]]
    ).reshape(-1, 1, 2)

    try:
        # aply homography to conners of src_img
        pts1_ = cv2.perspectiveTransform(pts1, H)
        pts = np.concatenate((pts1_, pts2), axis=0)

        # find max min of x,y coordinate
        [xmin, ymin] = np.int64(pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int64(pts.max(axis=0).ravel() + 0.5)
        t = [-xmin, -ymin]

        # top left point of image which apply homography matrix, which has x coordinate < 0, has side=left
        # otherwise side=right
        # source image is merged to the left side or right side of destination image
        if pts[0][0][0] < 0:
            side = "left"
            width_pano = width_dst + t[0]
        else:
            # width_pano = int(pts1_[3][0][0]) # x_max
            width_pano = int(xmax) # x_max
            side = "right"
        height_pano = ymax - ymin

        # Translation
        # https://stackoverflow.com/a/20355545
        Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
        src_img_warped = cv2.warpPerspective(src_img, Ht.dot(H), (width_pano, height_pano))
        # generating size of dst_img_rz which has the same size as src_img_warped
        dst_img_rz = np.zeros((height_pano, width_pano, 3))
        if side == "left":
            dst_img_rz[t[1]: height_src + t[1], t[0]: width_dst + t[0]] = dst_img
        else:
            dst_img_rz[t[1]: height_src + t[1], :width_dst] = dst_img

        # blending panorama
        pano = panorama_blending(dst_img_rz, src_img_warped, width_dst, side)

        # croping black region
        pano = crop(pano, height_dst, pts)
        return pano

    except BaseException:

        raise Exception("Please try again with another image set!")


def multi_stitching_superpoint(list_images):
    n = int(len(list_images) / 2 + 0.5)
    left = list_images[:n]
    right = list_images[n - 1 :]
    right.reverse()
    while len(left) > 1:
        dst_img = left.pop()
        src_img = left.pop()
        left_pano = warp_two_images_superpoint(src_img, dst_img)
        left_pano = left_pano.astype("uint8")
        left.append(left_pano)

    while len(right) > 1:
        dst_img = right.pop()
        src_img = right.pop()
        right_pano = warp_two_images_superpoint(src_img, dst_img)
        right_pano = right_pano.astype("uint8")
        right.append(right_pano)

    # if width_right_pano > width_left_pano, Select right_pano as destination. Otherwise is left_pano
    if right_pano.shape[1] >= left_pano.shape[1]:
        fullpano = warp_two_images_superpoint(left_pano, right_pano)
    else:
        fullpano = warp_two_images_superpoint(right_pano, left_pano)
    return fullpano


multi_stitching_superpoint(image_list[0], image_list[1])

full_panorama = multi_stitching_superpoint(image_list.copy())
full_panorama_normal = np.array(full_panorama, dtype=float)/float(255)
plt.figure(figsize=(15,10))
plt.imshow(full_panorama_normal)
plt.show()