# Age-Gender-and-Emotion-Detection-through-Images

Overview
This project aims to detect age, gender, and emotions from facial images using the DeepFace library and OpenCV for face detection. The repository includes code for analyzing static images and real-time video feed from a webcam.

Features
Age Detection: Predicts the age group of a person.
Gender Detection: Classifies the gender of a person.
Emotion Detection: Identifies the emotional state (e.g., happy, sad, angry, etc.) of a person.
Real-time Detection: Analyzes real-time video feed from a webcam.

Project Structure
images/: Contains sample images for testing.
notebooks/: Jupyter notebooks for exploratory data analysis and model training.
scripts/: Utility scripts for various tasks like image and video analysis.
results/: Directory for storing results and logs.

Installation
To get started with the project, clone the repository and install the required dependencies:
git clone https://github.com/yourusername/age-gender-emotion-detection.git
cd age-gender-emotion-detection
pip install -r requirements.txt

Usage
Analyzing a Static Image
Place your image in the images/ directory.
Run the script to analyze the image:
import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt

img_path = r"images/your_image.jpg"
img = cv2.imread(img_path)

# Analyze the image for age, gender, and emotion

results = DeepFace.analyze(img, actions=['age', 'gender', 'emotion'])

# Detect faces in the image

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

# Draw bounding boxes and annotate the image with results

for i, (x, y, w, h) in enumerate(faces):
cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
if i < len(results):
face_results = results[i]
age = face_results['age']
gender = face_results['dominant_gender']
emotion = face_results['dominant_emotion']
y_offset = y - 10
cv2.putText(img, f'Age: {age}', (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
cv2.putText(img, f'Gender: {gender}', (x, y_offset - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
cv2.putText(img, f'Emotion: {emotion}', (x, y_offset - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 10))
plt.imshow(img_rgb)
plt.axis('off')
plt.show()

Real-time Detection from Webcam
Run the script to start real-time detection:
import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
ret, frame = cap.read()
if not ret:
break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    results = DeepFace.analyze(frame, actions=['age', 'gender', 'emotion'], enforce_detection=False)

    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if i < len(results):
            face_results = results[i]
            age = face_results['age']
            gender = face_results['dominant_gender']
            emotion = face_results['dominant_emotion']
            y_offset = y - 10
            cv2.putText(frame, f'Age: {age}', (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, f'Gender: {gender}', (x, y_offset - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, f'Emotion: {emotion}', (x, y_offset - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

Dependencies
Python 3.7+
OpenCV
DeepFace
Matplotlib
Install the dependencies using:
pip install opencv-python-headless deepface matplotlib

Datasets
The project utilizes the DeepFace library which includes pretrained models for age, gender, and emotion detection. No additional datasets are required to run the provided code.

Results
Performance metrics and visualizations of the model results can be generated and stored in the results/ directory.

Contributions
Contributions to improve the project are welcome. Feel free to fork the repository and submit pull requests.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Contact
For any questions or issues, please open an issue in the repository or contact [saadatmalik268@gmail.com].
