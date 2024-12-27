import cv2
import numpy as np
import face_recognition
from datetime import datetime
from tensorflow.keras.models import model_from_json
import pandas as pd

# Load the emotion recognition model
def load_emotion_model(json_path, weights_path):
    with open(json_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(weights_path)
    return model

# Paths to emotion model files
emotion_model_json = "model_a1.json"  # Replace with the actual path
emotion_model_weights = "model_weights1.h5"  # Replace with the actual path

# Load the emotion recognition model
emotion_model = load_emotion_model(emotion_model_json, emotion_model_weights)

# Emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Load and encode the known face images
def load_and_encode_image(image_path):
    try:
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        return encoding
    except IndexError:
        print(f"No face found in the image: {image_path}")
        return None

# Paths to known face images
image_1_path = "sadface.jpg"  # Replace with the actual path
image_2_path = "sadface2.jpg"  # Replace with the actual path

# Generate encodings for the known faces
known_face_encodings = []
known_face_names = []

encoding_1 = load_and_encode_image(image_1_path)
if encoding_1 is not None:
    known_face_encodings.append(encoding_1)
    known_face_names.append("Student 1")

encoding_2 = load_and_encode_image(image_2_path)
if encoding_2 is not None:
    known_face_encodings.append(encoding_2)
    known_face_names.append("Student 2")

# Initialize attendance tracking
attendance = {}

def mark_attendance(name, emotion):
    if name not in attendance:
        now = datetime.now()
        time_string = now.strftime('%H:%M:%S')
        attendance[name] = {
            "Time": time_string,
            "Emotion": emotion
        }
        print(f"Marked attendance for {name} at {time_string} with emotion: {emotion}")

def process_video():
    video_capture = cv2.VideoCapture(0)  # Open webcam

    # Restrict the processing to a specific time interval
    start_time = datetime.strptime("09:30:00", "%H:%M:%S").time()
    end_time = datetime.strptime("10:00:00", "%H:%M:%S").time()

    while True:
        current_time = datetime.now().time()
        if not (start_time <= current_time <= end_time):
            print("System is inactive outside the scheduled time.")
            break

        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture video frame")
            break

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Detect faces and compute encodings
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                else:
                    name = "Unknown"
            else:
                name = "Unknown"

            # Extract face ROI for emotion detection
            top, right, bottom, left = [v * 4 for v in face_location]
            face_roi = frame[top:bottom, left:right]
            face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(face_gray, (48, 48))
            face_normalized = face_resized / 255.0
            face_input = np.expand_dims(face_normalized, axis=(0, -1))

            # Predict emotion
            emotion_prediction = emotion_model.predict(face_input, verbose=0)
            emotion_index = np.argmax(emotion_prediction)
            emotion_label = emotion_labels[emotion_index]

            # Mark attendance with emotion
            mark_attendance(name, emotion_label)

            # Draw rectangle and labels around the face
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, f"{name} - {emotion_label}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Display the video frame
        cv2.imshow('Video', frame)

        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Save attendance to a file
def save_attendance():
    df = pd.DataFrame.from_dict(attendance, orient='index')
    df.index.name = 'Name'
    df.to_csv("attendance.csv")
    print("Attendance data saved.")

if __name__ == "__main__":
    process_video()
    save_attendance()
