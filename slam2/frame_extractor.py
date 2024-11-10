import cv2
import os

def extract_frames(video_path, output_folder, every_n_frame=1):
    # Открываем видеофайл
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Ошибка открытия видео.")
        return

    # Создаем выходную папку, если её нет
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 0  # Счетчик кадров в видео
    extracted_count = 0  # Счетчик извлеченных кадров

    while True:
        # Читаем кадр из видео
        ret, frame = cap.read()
        if not ret:
            break  # Завершаем цикл, если кадры закончились

        # Сохраняем только каждый n-й кадр
        if frame_count % every_n_frame == 0:
            # Формируем имя файла для сохранения
            frame_filename = os.path.join(output_folder, f"frame_{extracted_count:04d}.JPG")
            # Сохраняем кадр как изображение в формате .jpg
            cv2.imwrite(frame_filename, frame)
            extracted_count += 1

        frame_count += 1

    # Освобождаем ресурсы
    cap.release()
    print(f"Извлечено {extracted_count} кадров из видео и сохранено в формате .jpg.")

# Укажите путь к вашему видеофайлу
video_path = "/home/skg/Projects/3D_ComputerVision/slam2/images/images.mkv"
# Укажите папку для сохранения кадров
output_folder = "/home/skg/Projects/3D_ComputerVision/slam2/images/frames"
# Извлекать каждый кадр или каждый n-й кадр (например, каждый пятый)
every_n_frame = 5

extract_frames(video_path, output_folder, every_n_frame)
