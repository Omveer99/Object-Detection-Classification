import cv2
import pytesseract

def extract_text(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("Error: Image not loaded!")
        return

    # Convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Optional: Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert binary image (if the text is white on black background)
    binary = cv2.bitwise_not(binary)

    # Perform OCR
    text = pytesseract.image_to_string(binary)

    return text

if __name__ == "__main__":
    pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'  # Ensure the correct Tesseract path
    image_path = '/Users/apple/Documents/project111/captured_image.jpg'  # Your image path
    extracted_text = extract_text(image_path)

    if extracted_text.strip():
        print("Extracted Text:", extracted_text)
    else:
        print("No text was extracted.")
