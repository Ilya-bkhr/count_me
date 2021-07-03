import os
import sys
import face_recognition
import cv2 as cv 
import csv
import numpy as np


def get_frame(path_to_video, frame_number):
    """Раскадровка"""
    cap = cv.VideoCapture(path_to_video) 
    count = 0 
    if not cap.isOpened():
        print("Не удается открыть видео")
        exit()
    while True:
        ret, frame = cap.read()
        if count % frame_number == 0:
            cv.imwrite(f'out_frames/reframe_{int((count + 1)/frame_number)}.jpg', frame)
        count += 1
        if not ret:
            print(f"Видео раскадированно")
            break
    cv.destroyAllWindows()


def person_on_frame():
    """Обнаружение человека на кадре, и его подсчеn"""
    if not os.path.exists('out_frames'):
        print('[ОШИБКА] не найдена папка с кадрами')
        sys.exit()

    known_enc = [] #enc = encoding
    images = os.listdir('out_frames')
    
    for(i, image) in enumerate(images):
        print(f'Обробатываемый кадр:[{i + 1}/{len(images)}]')
        face_img = face_recognition.load_image_file(f'out_frames/{image}')
        face_codes = face_recognition.face_encodings(face_img)
        print(f'На кадре обнаруженно:[{len(face_codes)}] лица')        
        if len(face_codes) == 0:
                print('На этом кадре НЕТ лиц')
                continue
        if not known_enc:
            known_enc.append(face_codes)
        else:
            for code in range(0, len(face_codes)):
                for item in range(0, len(known_enc)):
                    result = face_recognition.compare_faces(face_codes[code], known_enc[item])
                    if True in result:
                        print('Этот человек посчитан на предидущем кадре')
                    else:
                        known_enc[0].append(face_codes[code])   
    print(f'ОБНАРУЖЕННО: {len(known_enc[0])} людей')
    return known_enc[0] # Получаем результат в виде списка


def list_to_dict(enc_to_transform):
    """Преобразуем список в словарь"""
    transformed_enc = []
    for i in range(len(enc_to_transform)):
        transformed_enc.append({'id': i, 'face_code' : enc_to_transform[i]})
    return transformed_enc


def save_result(transformed_enc, path_to_csv):
    """"Сохраняем словарь в файл """
    csv_columns = ['id','face_code'] 
    with open(path_to_csv, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames = csv_columns, delimiter=';')
        writer.writeheader()
        for row in transformed_enc:
            writer.writerow(row)


def main():
    get_frame('video\example_2.mp4', 100)
    enc_to_transform = person_on_frame()
    transformed_enc = list_to_dict(enc_to_transform)
    save_result(transformed_enc, 'result.csv')


if __name__ == '__main__':
    main()

#