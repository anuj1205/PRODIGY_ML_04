from tensorflow.keras.models import load_model
from src.data_loader import load_data
from src.config import MODEL_PATH

def evaluate_model():
    print("📊 Evaluating model... please wait")

    X, y, _ = load_data()
    model = load_model(MODEL_PATH)
    loss, accuracy = model.evaluate(X, y)
    print(f"📊 Model Accuracy: {accuracy * 100:.2f}%")
