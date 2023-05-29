import os
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3,3),activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128,kernel_size=(3,3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Conv2D(128,kernel_size=(3,3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024,activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7,activation='softmax'))
emotion_model.load_weights('prj.h5')

emotion_dict = {0: 'Angry', 1: 'Sad', 2: 'Surprise', 3: 'Happy', 4: 'Neutral', 5: 'Disgusted', 6: 'Fear'}


cur_path = os.path.dirname(os.path.abspath(__file__))
emoji_dist={0:cur_path+"/emojis/angry.png",
            1:cur_path+"/emojis/sad.png", 
            2:cur_path+"/emojis/surprise.png",
            3:cur_path+"/emojis/happy.png", 
            4:cur_path+"/emojis/neutral.png",
            5:cur_path+"/emojis/disgust.png",
            6:cur_path+"/emojis/fear.png"}


cap = cv2.VideoCapture(1)


face_cascade = cv2.CascadeClassifier("C:/Users/KIIT/Desktop/Emojify/haar cascade. xml")

while True:
    
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    
    for (x, y, w, h) in faces:
        
        face = gray[y:y+h, x:x+w]

        
        face = cv2.resize(face, (48, 48))

        
        face = face.reshape(1, 48, 48, 1) / 255.0

                
        prediction = emotion_model.predict(face)
        emotion_label = np.argmax(prediction)
        emotion_emoji_path = emoji_dist[emotion_label]
        emoji_img = cv2.imread(emotion_emoji_path)

        
        emoji_img = cv2.resize(emoji_img, (h, w))
        frame[y:y+h, x:x+w] = emoji_img
        
    
    cv2.imshow('Emotion-Emoji',frame)
    
    
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

