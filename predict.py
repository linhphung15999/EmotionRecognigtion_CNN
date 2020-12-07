import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

from utils.plots import plot_one_box

model = model_from_json(open("model.json", "r").read())
model.load_weights('model.h5')
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def emotion_analysis(emotions):
    objects = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, emotions, align='center', alpha=0.9)
    plt.tick_params(axis='x', which='both', pad=10,width=4,length=10)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')
plt.show()



def cnn_predict(img):
    img = cv2.resize(img, (48, 48))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.array(255*img, np.uint8)
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x /= 255

    custom = model.predict(x)
    #print(custom[0])
    emotion_analysis(custom[0])

    m=0.000000000000000000001
    a=custom[0]
    s = 0.0
    for i in range(0,len(a)):
        s+=a[i]
        if a[i]>m:
            m=a[i]
            ind=i
        
    return([ind,np.round(m*1.0/s,2)])




##########################





