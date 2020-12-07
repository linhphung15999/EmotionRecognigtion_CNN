import cv2 as cv
import cv2
from PIL import Image

from utils.plots import plot_one_box

from opencv_functions import *
from utility_functions import *
from predict import *
from numpy import random

objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
colors = [[random.randint(0, 255) for _ in range(3)] for _ in objects]

faceCascades = load_cascades()

cv.namedWindow("preview")

# Open input video steam
vc = cv.VideoCapture(0)

# Check that video stream is running
if vc.isOpened(): # try to get the first frame
  rval, frame = vc.read()
  #frame = frame.astype(np.float32)
else:
  rval = False
i = 0
while rval:
  # Mirror image
  frame = np.fliplr(frame)
  # Detect faces
  detect = True
  if detect:
    # Find all faces
    with nostdout():
      _, faces = DetectFace(frame,True,faceCascades,single_face=False,second_pass=False,draw_rects=False,scale=1.0)
    #frame = addEmoji(frame,faces,emoji)
    if len(faces) == 0 or faces is None:
      # No faces found
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
          resu = cnn_predict(img)
          ind = int(resu[0])
          label = objects[ind] + str(resu[1])
          plot_one_box([x,y,x+w,y+h], frame, label=label, color=[colors[ind][0]/255.0,colors[ind][1]/255.0,colors[ind][2]/255.0], line_thickness=2)
    

  # Show video with faces
  
  cv.imshow("preview", frame)
  # Read in next frame
  rval, frame = vc.read()

  # Wait for user to press key. On ESC, close program
  key = cv.waitKey(20)
  if key == 27: # exit on ESC
    break
  elif key == 115 or key == 83: # ASCII codes for s and S
    filename = saveTestImage(img,outDir="output")
    print ("Image saved to ./" + filename)

cv.destroyWindow("preview")


