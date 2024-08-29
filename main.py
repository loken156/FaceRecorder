from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import joblib

# Load the face detection model
face_classifier = cv2.CascadeClassifier(r'C:\Users\Jontes\repos\ai\FaceRec\haarcascade_frontalface_default.xml')

# Load the emotion detection model
emotion_classifier = load_model(r'C:\Users\Jontes\repos\ai\FaceRec\emotion_model_sklearn.keras')

# Load the gender detection model
gender_classifier = load_model(r'C:\Users\Jontes\repos\ai\FaceRec\gender_model_sklearn.keras')

# Load the saved scaler
scaler = joblib.load('age_scaler.pkl')
# Load the age detection model
age_classifier = load_model(r'C:\Users\Jontes\repos\ai\FaceRec\age_model_best.keras')

# Define emotion and gender labels
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
gender_labels = ['Female', 'Male']

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]  # For age prediction (color image)
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Check if the region of interest has enough data
        if np.sum([roi_gray]) != 0:
            # Prepare image for emotion prediction
            roi_gray_resized = roi_gray.astype('float') / 255.0
            roi_gray_resized = img_to_array(roi_gray_resized)
            roi_gray_resized = np.expand_dims(roi_gray_resized, axis=0)

            # Emotion prediction
            emotion_prediction = emotion_classifier.predict(roi_gray_resized)[0]
            emotion_label = emotion_labels[emotion_prediction.argmax()]

            # Convert to grayscale for gender prediction
            roi_gender = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
            roi_gender = cv2.resize(roi_gender, (48, 48), interpolation=cv2.INTER_AREA)
            roi_gender = roi_gender.astype('float') / 255.0
            roi_gender = img_to_array(roi_gender)
            roi_gender = np.expand_dims(roi_gender, axis=-1)  # Add this line to ensure the shape is (48, 48, 1)
            roi_gender = np.expand_dims(roi_gender, axis=0)  # Shape will be (1, 48, 48, 1)

            # Gender prediction
            gender_prediction = gender_classifier.predict(roi_gender)[0]
            gender_label = gender_labels[gender_prediction.argmax()]

            # Prepare image for age prediction (if the age model expects a color image)
            roi_color_resized = cv2.resize(roi_color, (48, 48), interpolation=cv2.INTER_AREA)
            roi_color_resized = roi_color_resized.astype('float') / 255.0
            roi_color_resized = img_to_array(roi_color_resized)
            roi_color_resized = np.expand_dims(roi_color_resized, axis=0)

            # Age prediction
            age_prediction = age_classifier.predict(roi_color_resized)[0]
            print("Raw model output for age prediction:", age_prediction)

            # Inverse transform to get the actual age
            age_actual = scaler.inverse_transform(age_prediction.reshape(-1, 1)).flatten()
            print("Predicted age:", age_actual)

            # Convert the predicted age to an integer for display
            age_label = int(age_actual[0])

            # Display emotion, gender, and age labels on the frame
            label_position = (x, y)
            cv2.putText(frame, f'{gender_label}, {emotion_label}, Age: {age_label}', label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Emotion, Gender & Age Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
