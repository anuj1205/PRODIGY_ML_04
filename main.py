import os
import warnings
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict_image


def main():
    print("🖐️ Hand Gesture Recognition System")
    print("1. Train Model")
    print("2. Evaluate Model")
    print("3. Predict Gesture from Image")

    choice = input("Enter choice (1/2/3): ")

    if choice == "1":
        train_model()
    elif choice == "2":
        evaluate_model()
    elif choice == "3":
        image_path = input("Enter image path: ")
        predict_image(image_path)
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()
