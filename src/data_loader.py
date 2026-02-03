import os
import cv2
import numpy as np
from src.config import IMG_SIZE, DATASET_PATH

def load_data():
    images = []
    labels = []
    label_map = {}
    label_id = 0

    for subject in os.listdir(DATASET_PATH):
        subject_path = os.path.join(DATASET_PATH, subject)
        if not os.path.isdir(subject_path):
            continue

        for gesture in os.listdir(subject_path):
            gesture_path = os.path.join(subject_path, gesture)
            if not os.path.isdir(gesture_path):
                continue

            if gesture not in label_map:
                label_map[gesture] = label_id
                label_id += 1

            for img_name in os.listdir(gesture_path):
                img_path = os.path.join(gesture_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                labels.append(label_map[gesture])

    images = np.array(images, dtype="float32") / 255.0
    images = images.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    labels = np.array(labels)

    return images, labels, label_map
    