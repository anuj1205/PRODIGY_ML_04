from src.data_loader import load_data
from src.model import build_model
from src.config import EPOCHS, MODEL_PATH
from sklearn.model_selection import train_test_split
import os
import pickle


def train_model():
    print("📂 Loading dataset... please wait")
    
    X, y, label_map = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_model(X.shape[1:], len(label_map))

    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        validation_data=(X_test, y_test)
    )

    os.makedirs("model", exist_ok=True)
    model.save(MODEL_PATH)
    
    with open("model/label_map.pkl", "wb") as f:
        pickle.dump(label_map, f)


    print("✅ Model trained and saved successfully")
