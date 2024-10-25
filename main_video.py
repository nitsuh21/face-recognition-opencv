import cv2
from simple_facerec import SimpleFacerec

# Initialize the SimpleFacerec class
sfr = SimpleFacerec()

# Load known face encodings from images in the 'images' directory
sfr.load_encoding_images("images/")  # Replace 'images/' with your directory path

# Start video capture from the camera
video_capture = cv2.VideoCapture(0)

while True:
    # Capture a single frame from the camera
    ret, frame = video_capture.read()

    # Check if the frame is captured successfully
    if not ret:
        print("Failed to capture frame. Exiting.")
        break

    # Detect known faces in the frame
    face_locations, face_names = sfr.detect_known_faces(frame)

    # Draw boxes around faces and add names
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Display the name below the face
        cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display the resulting frame
    cv2.imshow("Face Recognition", frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
video_capture.release()
cv2.destroyAllWindows()
