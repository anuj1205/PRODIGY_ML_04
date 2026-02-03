import os
import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from src.config import MODEL_PATH, IMG_SIZE

def predict_image(image_path):
    image_path = image_path.strip().strip('"').replace("\\", "/")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ File does not exist: {image_path}")

    model = load_model(MODEL_PATH)

    # Load label map
    with open("model/label_map.pkl", "rb") as f:
        label_map = pickle.load(f)

    # Reverse mapping: {id: name}
    id_to_label = {v: k for k, v in label_map.items()}

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    prediction = model.predict(img)
    class_id = int(np.argmax(prediction))
    gesture_name = id_to_label[class_id]

    print(f"🖐️ Predicted Gesture: {gesture_name}")
