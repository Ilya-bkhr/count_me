import csv
import os
import sys

import cv2 as cv

import face_recognition

import matplotlib.pyplot as plt


def get_frame(path_to_video, frame_number):
    """Decoding video"""
    cap = cv.VideoCapture(path_to_video)
    count = 0
    if not cap.isOpened():
        print("There is no way to open selected file")
        exit()
    while True:
        ret, frame = cap.read()
        if count % frame_number == 0:
            cv.imwrite(f'out_frames/reframe_{int((count + 1)/frame_number)}.jpg', frame)
        count += 1
        if not ret:
            print(f'Video have decoded')
            break
    cv.destroyAllWindows()


def check_files():
    """Checking the directory with frames"""
    if not os.path.exists('out_frames'):
        print("[ERORR] couldn't search directory with frames")
        sys.exit()


def person_on_frame():
    """Detection persons on frame and counting detected persons"""
    known_enc = []  # enc = encoding
    images = os.listdir('out_frames')
    list_per_second = []
    person_per_second = []

    for(i, image) in enumerate(images):
        print(f'Processing frame:[{i + 1}/{len(images)}]...')
        face_img = face_recognition.load_image_file(f'out_frames/{image}')
        face_codes = face_recognition.face_encodings(face_img) # Reading the encoding from the frame
        print(f'Detected:[{len(face_codes)}] face(s) on frame')
        list_per_second.append(i)
        person_per_second.append(len(face_codes))

        if len(face_codes) == 0:
                print('NO face(s) on frame')
                continue
        if not known_enc:
            known_enc.append(face_codes)
        else:
            for code in range(0, len(face_codes)):
                for item in range(0, len(known_enc)):
                    result = face_recognition.compare_faces(face_codes[code], known_enc[item])
                    if True in result:
                        print('That person was counted on previous frame')
                    else:
                        known_enc[0].append(face_codes[code])
    print(f'DETECTED: {len(known_enc[0])} person(s)')
    return known_enc[0], list_per_second, person_per_second  # List of uniq person was created


def list_to_dict(enc_to_transform):
    """Tranforming list in dict"""
    transformed_enc = []
    for i in range(len(enc_to_transform)):
        transformed_enc.append({'id': i, 'face_code': enc_to_transform[i]})
    return transformed_enc


def save_result(transformed_enc, path_to_csv):
    """"Saving the result in csv file"""
    csv_columns = ['id', 'face_code']
    with open(path_to_csv, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_columns, delimiter=';')
        writer.writeheader()
        for row in transformed_enc:
            writer.writerow(row)


def show_result(person_per_second, list_per_second, enc_to_transform):
    """"Drawing of chart"""
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.stackplot(list_per_second, person_per_second, labels=['Number of person'])
    ax.set_title(f'in total: {len(enc_to_transform)} person(s) on video')
    ax.legend(loc='upper left')
    ax.set_ylabel('Person')
    ax.set_xlabel('Time')
    ax.set_xlim(xmax=list_per_second[-1], xmin=list_per_second[0])
    fig.tight_layout()
    plt.savefig('Test_chart.png')


def main():
    get_frame('video\example_6.mp4', 33)
    check_files()
    enc_to_transform, list_per_second, person_per_second = person_on_frame()
    transformed_enc = list_to_dict(enc_to_transform)
    save_result(transformed_enc, 'result.csv')
    show_result(person_per_second, list_per_second, enc_to_transform)


if __name__ == '__main__':
    main()
