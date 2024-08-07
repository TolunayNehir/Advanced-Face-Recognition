import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

recognizer = cv2.face.LBPHFaceRecognizer_create()
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def load_labels_from_file(file_path):
    with open(file_path, 'r') as file:
        labels = [int(line.strip()) for line in file.readlines()]
    return labels

def load_classes_from_file(classes_path):
    with open(classes_path, 'r') as file:
        classes = [line.strip() for line in file.readlines()]
    return classes

def train_recognizer(imagesfolder, labelsfile, haar_cascade):
    images = []
    labels = []

    face_ids = load_labels_from_file(labelsfile)

    for idx, filename in enumerate(os.listdir(imagesfolder)):
        img_path = os.path.join(imagesfolder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]
                images.append(face)
                labels.append(face_ids[idx])
        else:
            print(f"Image at {img_path} could not be read.")

    if len(images) == 0 or len(labels) == 0:
        print("No faces or labels found. Training aborted.")
        return

    recognizer.train(images, np.array(labels))
    recognizer.save('face_trained.yml')
    print("Training completed and model saved as 'face_trained.yml'.")

def recognize_faces(face_recognizer, haar_cascade, classes_path):
    face_recognizer.read('face_trained.yml')
    classes = load_classes_from_file(classes_path)
    print("Classes:", classes)

    while True:
        test_img_path = input("Image for recognition (for exit press q): ")

        if test_img_path.lower() == 'q':
            print("Exiting.")
            break

        test_img = cv2.imread(test_img_path)
        if test_img is None:
            print("Image cannot be read, wrong path or file name.")
            continue

        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        faces = haar_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            if face.size == 0:
                print("Face region is empty.")
                continue

            label, confidence = recognizer.predict(face)
            cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(test_img, f'Class: {classes[label]}, Conf: {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
        plt.show()

print("Welcome to the Advanced Face Recognition")
choice = input("1 Train Recognizer, 2 Recognize Faces: ")

if int(choice) == 1:
    imagesfolder = input("Images folder path: ")
    labelsfile = input("Labels file path: ")
    train_recognizer(imagesfolder, labelsfile, haar_cascade)

elif int(choice) == 2:
    classes_path = input("Classes file path: ")
    recognize_faces(recognizer, haar_cascade, classes_path)
    
else:
    print("Wrong Choice")
