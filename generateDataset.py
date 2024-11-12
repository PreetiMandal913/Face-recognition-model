import cv2
import os

# 1. Generate the dataset
# 2. Train the classifier and save it
# 3. Detect the face and name it if it is already stored in our dataset

def generate_dataset():
    # Get the current directory of the script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the full path to the cascade file
    cascade_path = os.path.join(current_dir, 'haarcascade_frontalface_default.xml')

    # Load the cascade
    face_classifier = cv2.CascadeClassifier(cascade_path)
    
    # Create data directory if it does not exist
    data_dir = os.path.join(current_dir, 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y+h, x:x+w]
            return cropped_face  # Return the first detected face

    cap = cv2.VideoCapture(0)
    id = 3
    img_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        cropped_face = face_cropped(frame)
        if cropped_face is not None:
            img_id += 1
            face = cv2.resize(cropped_face, (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name_path = os.path.join(data_dir, f"user.{id}.{img_id}.jpg")
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Cropped face", face)

            # Exit on 'Enter' key or after collecting 200 samples
            if cv2.waitKey(1) == 13 or img_id == 200:
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Collecting samples is completed.")

generate_dataset()
