import sys
import os
import face_recognition
import cv2 as cv 
import csv
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


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
    list_per_second = [] 
    person_per_second = [] 

    for(i, image) in enumerate(images):
        print(f'Обробатываемый кадр:[{i + 1}/{len(images)}]')
        face_img = face_recognition.load_image_file(f'out_frames/{image}')
        face_codes = face_recognition.face_encodings(face_img)
        print(f'На кадре обнаруженно:[{len(face_codes)}] лица')
        list_per_second.append(i)
        person_per_second.append(len(face_codes))

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
                        print('Этот человек посчитан на предЫдущем кадре')
                    else:
                        known_enc[0].append(face_codes[code])   
    print(f'ОБНАРУЖЕННО: {len(known_enc[0])} человека')
    return known_enc[0], list_per_second, person_per_second # Получаем результат в виде списка



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


def show_result(person_per_second, list_per_second, enc_to_transform):
    """"Рисуем график"""
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.stackplot(list_per_second, person_per_second, labels =['Количество человек'])
    ax.set_title(f'Всего на видео {len(enc_to_transform)} человек(а)')
    ax.legend(loc='upper left')
    ax.set_ylabel('Люди')
    ax.set_xlabel('Время')
    ax.set_xlim(xmax= list_per_second[-1], xmin = list_per_second[0])
    fig.tight_layout()
    plt.savefig('Test_chart.png')


def main():
    get_frame('video\example_5.mp4', 33)
    enc_to_transform, list_per_second, person_per_second = person_on_frame()
    transformed_enc = list_to_dict(enc_to_transform)
    save_result(transformed_enc, 'result.csv')
    show_result(person_per_second, list_per_second, enc_to_transform)

if __name__ == '__main__':
    main()
