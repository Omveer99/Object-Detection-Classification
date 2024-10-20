import cv2
import numpy as np
from PIL import Image

def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalize brightness and contrast
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # Apply Gaussian blur to remove noise
    blurred = cv2.GaussianBlur(normalized, (5, 5), 0)

    # Apply edge detection
    edges = cv2.Canny(blurred, 100, 200)

    # Save preprocessed image
    cv2.imwrite('preprocessed_image.jpg', edges)

    return edges

# Example usage
preprocessed_image = preprocess_image('captured_image.jpg')
