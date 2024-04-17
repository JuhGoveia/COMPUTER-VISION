# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 16:53:31 2024

@author: Emil Freme
"""
import cv2
import dlib
import easygui
import numpy as np
import os
import time
#import mediapipe as mp
#from mediapipe.tasks import python
#from mediapipe.tasks.python import vision
#from mediapipe import solutions
#from mediapipe.framework.formats import landmark_pb2

def distance3(a, b):
    import math
    return math.sqrt( (a.x - b.x)**2 + (a.y - b.y)**2 )

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

class LivenessDetection:
    def __init__(self):
        self.last_right_EAR = 0
        self.last_left_EAR  = 0
        self.treshold       = 3
        self.isLive         = False

        self.landmark_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
#        base_options_landmarks = python.BaseOptions(model_asset_path="face_landmarker.task")
#        options_landmarks = vision.FaceLandmarkerOptions(
#                base_options=base_options_landmarks,
#                min_tracking_confidence=0.6,
#                min_face_presence_confidence=0.6,
#                min_face_detection_confidence=0.6,
#                num_faces=1)
#
#        self.landmark_detector = vision.FaceLandmarker.create_from_options(options_landmarks)
#        print(self.landmark_detector)

    def check(self, image):
        np_image = np.array(image)
        height, width = image.shape[:2]
        full_image_rect = dlib.rectangle(0, 0, width - 1, height - 1)
        #mp_frame_l = mp.Image(image_format=mp.ImageFormat.SRGB, 
        #                      data=np_image)
        result_landmark = self.landmark_detector(np_image, full_image_rect)
        #self.landmark_detector.detect(
        #        mp_frame_l)
        
        

        self.update_ears(result_landmark)
        pass

    def eye_aspect_ratio(self, eye_landmakrs):
        TOP = 0
        BOTTOM = 1
        OUTER = 2
        INNER = 3
        vertical_distance = distance3(eye_landmakrs[TOP], eye_landmakrs[BOTTOM])
        horizontal_distance = distance3(eye_landmakrs[OUTER], eye_landmakrs[INNER])
        print(f"{horizontal_distance/vertical_distance}")
        return horizontal_distance / vertical_distance 

    def update_ears(self, landmarks_result):

        l_eyes_ids = [37, 41, 36, 39]
        r_eyes_ids = [43, 47, 45, 42]

        left_eye_landmarks = [landmarks_result.part(i) for i in l_eyes_ids]
        right_eye_landmarks = [landmarks_result.part(i) for i in r_eyes_ids]
        

        new_left_EAR = self.eye_aspect_ratio(left_eye_landmarks)
        new_right_EAR = self.eye_aspect_ratio(right_eye_landmarks)

        if self.last_left_EAR == 0 or self.last_right_EAR == 0:
            self.last_left_EAR = new_left_EAR
            self.last_right_EAR = new_right_EAR
            return;

        if(abs(new_left_EAR - self.last_left_EAR) > self.treshold or
            abs(new_right_EAR - self.last_right_EAR) > self.treshold):

            self.isLive = True



def new_face():
    haar_file = 'haarcascade_frontalface_default.xml'
    datasets = 'captures'

    # Initialize face cascade classifier
    face_cascade = cv2.CascadeClassifier(haar_file)
    webcam = cv2.VideoCapture(0)
    count = -1
    person_name = easygui.enterbox("Nome do Usuário")
    while count < 50:
        (_, im) = webcam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 4)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (130, 100))

            # Create a folder for each recognized face
            person_folder = os.path.join(datasets, person_name)
            if not os.path.isdir(person_folder):
                os.mkdir(person_folder)

            # Count existing images in the folder
            image_count = len(os.listdir(person_folder))

            le_time = time.time()

            # Save the face as an image in the person's folder
            image_path = os.path.join(person_folder, f"{le_time + 1}.png")
            cv2.imwrite(image_path, face_resize)
            count += 1

        cv2.imshow('OpenCV', im)
        key = cv2.waitKey(10)

        if key == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()
    pass

def run_recognition():
    liveness_detection = LivenessDetection()
    size = 4
    haar_file = 'haarcascade_frontalface_default.xml'
    datasets = 'captures'
    
    # Create a list of names from subdirectories in 'fotos' folder
    names = [name for name in os.listdir(datasets) if os.path.isdir(os.path.join(datasets, name))]
    
    # Initialize the face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + haar_file)
    
    # Define the width and height for resizing
    width, height = 130, 100
    
    # Create a LBPH Face Recognizer
    model = cv2.face.LBPHFaceRecognizer.create()
    
    # Create a list to store training data
    training_data = []
    
    # Load and preprocess training images
    for name in names:
        folder_path = os.path.join(datasets, name)
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (width, height))
            training_data.append((image, names.index(name)))
    
    # Train the model
    labels = [label for (_, label) in training_data]
    images = [image for (image, _) in training_data]
    model.train(images, np.array(labels))
    
    # Save the trained model to a file
    model.save('trained_model.xml')
    
    # Load the trained model
    model.read('trained_model.xml')
    
    # Open the webcam
    webcam = cv2.VideoCapture(0)
    frame = 1
    while True:
        _, frame = webcam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (width, height))
            
            
            
            # Try to recognize the face
            prediction = model.predict(face_resize)

            if prediction[1] < 500:
                name = names[prediction[0]]
                confidence = int(100 * (1 - (prediction[1]) / 300))
                if confidence > 75:
                    liveness_detection.check(face_resize)
                    cv2.putText(frame, f"{name} Is Live:{liveness_detection.isLive}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, str(confidence) + '%', (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, 'Quem es tu?', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, 'Nao te conheço!', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Detectando Faces', frame)

        if cv2.waitKey(16) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()
    pass


def main():
    option = easygui.choicebox("Selecione uma operação", "", ["Novo Cadastro", "Rodar Reconhecimento"])
    
    if option == "Novo Cadastro":
        new_face()
        
    if option == "Rodar Reconhecimento":
        run_recognition()
    
    pass

if __name__ == "__main__":
    main()
