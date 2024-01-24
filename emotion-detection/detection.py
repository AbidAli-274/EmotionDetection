import streamlit as st
import cv2
import numpy as np
from keras.models import model_from_json
import tempfile
import os

# Load the emotion detection model
json_file = open("facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("facialemotionmodel.h5")

# Load the face cascade
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to extract features from the image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def detect_emotions_in_frame(frame):
    labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize the frame for faster processing
    resized_frame = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
    
    faces = face_cascade.detectMultiScale(resized_frame, 1.3, 5)
    try:
        for (p, q, r, s) in faces:
            face_image = resized_frame[q:q + s, p:p + r]
            cv2.rectangle(frame, (2*p, 2*q), (2*(p + r), 2*(q + s)), (255, 0, 0), 2)
            face_image = cv2.resize(face_image, (48, 48))
            img = extract_features(face_image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            cv2.putText(frame, '%s' % (prediction_label), (2*p - 10, 2*q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))
        return frame
    except cv2.error:
        return frame

def main():
    st.title("Facial Emotion Recognition on Video")

    uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])

    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
        st.write(file_details)

        file_bytes = uploaded_file.read()

        if uploaded_file.type.startswith('video'):
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(file_bytes)

            cap = cv2.VideoCapture(temp_file.name)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            stframe = st.image([])
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process every alternate frame for faster display
                if cap.get(1) % 2 == 0:
                    detected_frame = detect_emotions_in_frame(frame)
                    stframe.image(detected_frame, channels="BGR")

            temp_file.close()
            os.unlink(temp_file.name)

            cap.release()

    st.text("Upload a video file to start facial emotion detection.")

if __name__ == "__main__":
    main()
