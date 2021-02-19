import os
from datetime import datetime
import cv2 as cv

names = {1: "Demming"}  # dictionary of users with access
cascade_path = 'C:/Users/austi/anaconda3/Lib/site-packages/cv2/data/'  # path for haarcascade folder
face_detector = cv.CascadeClassifier(cascade_path + 'haarcascade_frontalface_default.xml')  # facial recognition file

recognizer = cv.face.LBPHFaceRecognizer_create()  # instantiate face recognizer
recognizer.read('lbph_trainer.yml')  # read .yml file of trained face

test_path = './demming_tester'  # path for tester image folder
image_paths = [os.path.join(test_path, f) for f in os.listdir(test_path)]  # paths for images in tester folder

for image in image_paths:  # loop through image_paths list
    predict_image = cv.imread(image, cv.IMREAD_GRAYSCALE)  # create grayscale image
    faces = face_detector.detectMultiScale(predict_image, scaleFactor=1.05, minNeighbors=5)  # detect faces

    for (x, y, h, w) in faces:  # for coords in face rectangle
        print(f"\nAccess requested at {datetime.now()}.")  # post message to terminal
        face = cv.resize(predict_image[y:y + h, x:x + w], (100, 100))  # resize image
        predicted_id, dist = recognizer.predict(face)  # variables for id orediction and distance
        if predicted_id == 1 and dist <= 95:  # if predicted id is demming and he is under 95 away
            name = names[predicted_id]  # retrieve registered users name
            print("{} identified as {} with distance={}".format(image, name, round(dist, 1)))  # print message to log
            print(f"Access granted to {name} at {datetime.now()}.")  # print message to log
        else:  # otherwise
            name = "unknown"  # set name to unknown
            print(f"{image} is {name}.")  # print message to terminal
        cv.rectangle(predict_image, (x, y), (x + w, y + h), 255, 2)  # draw rectangle around face
        cv.putText(predict_image, name, (x + 1, y + h - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)  # put text on image
        cv.imshow('ID', predict_image)  # show image
        cv.waitKey(2000)  # wait 2 seconds
        cv.destroyAllWindows()  # close all windows
