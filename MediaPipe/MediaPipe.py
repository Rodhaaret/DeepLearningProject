import mediapipe as mp
import cv2
import numpy as np
import uuid 
import os 
import time

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


def extract_bounding_box(image,detection):
    width = image.shape[1]
    height = image.shape[0]
    
    if detection.score[0] > 0.80:
        boundBox = detection.location_data.relative_bounding_box

        x = int(boundBox.xmin * width)
        w = int(boundBox.width * width)

        y = int(boundBox.ymin * height)
        h = int(boundBox.height * height)
    return x,w,y,h

pTime = 0
count = 0
# For webcam input:
vidcap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.6) as face_detection:
  while vidcap.isOpened():

    cTime = time.time()
    fps= 1 / (cTime - pTime)
    pTime = cTime

    success, image = vidcap.read()

    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    cv2.putText(image, f'FPS: {int(fps)}', (20,50),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)

    image.flags.writeable = True
    results = face_detection.process(image)

    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:
      for detection in results.detections:

        x,w,y,h = extract_bounding_box(image,detection)
        if(x != None):
          face_size = image[y:y+h,x:x+w]

          face_img = cv2.resize(face_size, (224,224), interpolation = cv2.INTER_AREA)
          mp_drawing.draw_detection(image, detection)
          #cv2.imwrite('C:/Users/Thoma/OneDrive/Desktop/UNI/E22/Deep Neural Networks/dnn/DeepLearningProject/ournet/face_thomas/thomas'+ str(count)+ '.png', face_img)
        count += 1

    cv2.imshow(' Face Image', face_img)
    cv2.imshow('MediaPipe Face Detection', image)

    if cv2.waitKey(5) & 0xFF == 27:
      break

vidcap.release()
cv2.destroyAllWindows()