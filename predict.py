import numpy as np
import cv2
from tensorflow.keras.models import load_model # type: ignore

def load_and_prepare_image(image_path, img_size=(64, 64)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, img_size) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_image(image_path, model_path):
    model = load_model(model_path)
    img = load_and_prepare_image(image_path)

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)

    return predicted_class

# Example usage
image_path = '/path/to/new/image.jpg'  # Replace with your image path
model_path = 'classification_model.h5'
predicted_class = predict_image(image_path, model_path)
print(f"Predicted Class Index: {predicted_class[0]}")
