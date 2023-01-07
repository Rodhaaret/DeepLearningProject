from torch import nn, optim, as_tensor
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.nn.init import *
from torchvision import transforms, utils, datasets, models
# from models.inception_resnet_v1 import InceptionResnetV1
from torchvision.models.inception import Inception_V3_Weights

import cv2 as cv2
from PIL import Image
from pdb import set_trace
import time
import copy
from pathlib import Path
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage import io, transform
from tqdm import trange, tqdm
import csv
import glob
import dlib
import pandas as pd
import numpy as np
import os


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
vidcap = cv2.VideoCapture(0)

old = [0,0,0,0]
count = 0
while True:
    ret, frame = vidcap.read()
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
    if len(face) == 0:
        print("No face detected")
        # x, y, w, h = old
    else:
        x, y , w, h = face[0]
        roi_color = frame[y:y+h,x:x+w]
        resized = cv2.resize(roi_color, (224,224), interpolation = cv2.INTER_AREA)
        cv2.imshow('cropped', resized)
        cv2.rectangle(frame,(x,y), (x+w,y+h),(0,255,0),2)
        cv2.imshow('frame', frame)
        cv2.imwrite('C:/Users/Thoma/OneDrive/Desktop/UNI/E22/Deep Neural Networks/dnn/DeepLearningProject/ournet/face_thomas/thomas'+ str(count)+ '.png', resized)
        count += 1
    if len(face) != 0:
        old = face[0]
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vidcap.release()
cv2.destroyAllWindows()
