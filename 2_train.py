import os
import numpy as np
import cv2 as cv

cascade_path = 'C:/Users/austi/anaconda3/Lib/site-packages/cv2/data/'  # path for haarcascade files folder
face_detector = cv.CascadeClassifier(cascade_path + 'haarcascade_frontalface_default.xml')  # cascade xml file for faces

train_path = './demming_trainer'  # use for provided demming face
image_paths = [os.path.join(train_path, f) for f in os.listdir(train_path)]  # directory of training images
images, labels = [], []  # lists for images and labels

for image in image_paths:  # for each image file
    train_image = cv.imread(image, cv.IMREAD_GRAYSCALE)  # create grayscale copy
    label = int(os.path.split(image)[-1].split('.')[1])  # extract numerical label from filename
    name = os.path.split(image)[-1].split('.')[0]  # extract name from filename
    frame_num = os.path.split(image)[-1].split('.')[2]  # extract frame number from filename
    faces = face_detector.detectMultiScale(train_image)  # create numpy arrays for trainer image
    for (x, y, w, h) in faces:  # for face coords rectangle in the faces array
        images.append(train_image[y:y +h, x:x + w])  # append face rectangle to list of faces
        labels.append(label)  # append label to labels list
        print(f"Preparing training images for {name}.{label}.{frame_num}")  # print status message to user
        cv.imshow("Training Image", train_image[y:y + h, x:x + w])  # show image on screen
        cv.waitKey(50)  # wait 50 milliseconds

    cv.destroyAllWindows()  # close all windows

    recognizer = cv.face.LBPHFaceRecognizer_create()  # instantiate the recognizer object
    recognizer.train(images, np.array(labels))  # pass recognizer the images and labels arrays
    recognizer.write('lbph_trainer.yml')  # create file for trainer results
    print("Training complete. Exiting...")  # print status to user


