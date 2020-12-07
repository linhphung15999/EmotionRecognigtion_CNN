import cv2 as cv
import cv2
from PIL import Image

from utils.plots import plot_one_box

from opencv_functions import *
from utility_functions import *
from predict import *
import os
from tqdm import tqdm
from numpy import random

objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
colors = [[random.randint(0, 255) for _ in range(3)] for _ in objects]

faceCascades = load_cascades()

od = os.getcwd()+"/test_imgs"
sd = os.getcwd()+"/output"
	
for ims in tqdm(os.listdir(od)):
        ori_path = od+"/"+ims
        save_path = sd + "/"+ims
        frame = cv2.imread(ori_path)
        o_f = frame
        _, faces = DetectFace(frame,True,faceCascades,single_face=False,second_pass=False,draw_rects=False,scale=1.0)
        if len(faces) == 0 or faces is None:
            pass
        else:
        # Get a label for each face
        #labels = classify_video_frame(frame, faces, VGG_S_Net, categories=None)
      
        # Add an emoji for each label
        #frame = addMultipleEmojis(frame,faces,emojis,labels)
        
        # Print first emotion detected
        #print categories[labels[0]]
            frame = frame.astype(np.float32)
            frame /= 255.0

      
            for x,y,w,h in faces:
                img = frame[y:y+h,x:x+w,:]
          # Input image should be WxHxK, e.g. 490x640x3
          #cv.imshow("preview", img)
                res = cnn_predict(img)
                ind = int(res[0])
                label = objects[ind] + str(res[1])
                plot_one_box([x,y,x+w,y+h], o_f, label=label, color=colors[ind], line_thickness=2)
        
        cv2.imwrite(save_path,o_f)

        

