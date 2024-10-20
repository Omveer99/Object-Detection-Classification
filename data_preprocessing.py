import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_data(data_dir, img_size=(64, 64)):
    images = []
    labels = []

    # Iterate through the directories
    for label in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, label)
        if os.path.isdir(folder_path):
            for img_file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, img_size)  # Resize image
                images.append(img)
                labels.append(label)

    # Convert to NumPy arrays
    X = np.array(images, dtype='float32') / 255.0  # Normalize images
    y = np.array(labels)

    # Convert labels to numerical format
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    return X, y_categorical, le.classes_

# Example usage
data_dir = '/Users/apple/Documents/project111/dataset'  # Replace with your dataset path
X, y, class_names = load_data(data_dir)
print(f"Loaded {len(X)} images belonging to {len(class_names)} classes.")
