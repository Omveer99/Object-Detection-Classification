import cv2

# Initialize camera (0 for default webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Display the frame
    cv2.imshow('Live Camera Feed', frame)

    # Save the image when 's' is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('captured_image.jpg', frame)
        print("Image saved!")
        break

    # Exit the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
