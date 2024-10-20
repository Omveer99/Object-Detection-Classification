import cv2

def detect_objects(image_path, cascade_path):
    print("Starting object detection...")

    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not loaded!")
        return

    cascade = cv2.CascadeClassifier(cascade_path)

    if cascade.empty():
        print("Error: Cascade file not loaded correctly!")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("Image converted to grayscale.")

    objects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(objects) == 0:
        print("No objects detected.")
    else:
        print(f"Detected {len(objects)} object(s).")

    for (x, y, w, h) in objects:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Save the result to file
    cv2.imwrite('detected_objects.jpg', img)
    print("Object detection complete, image saved as 'detected_objects.jpg'.")


    # # Instead of displaying, save the image
    # cv2.imwrite('detected_objects.jpg', img)
    # # Save the resulting image with detected objects
    # cv2.imwrite('detected_objects.jpg', img)

    # # Display the output
    # cv2.imshow('Detected Objects', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# Example usage
cascade_path = '/Users/apple/Documents/project111/haarcascade_frontalface_default.xml'  # Absolute path to your cascade XML file
detect_objects('/Users/apple/Documents/project111/captured_image.jpg', cascade_path)  # Absolute path to your image

