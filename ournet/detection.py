import torch
from torch import nn, optim, as_tensor
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.nn.init import *
from torchvision import transforms, utils, datasets, models
from PIL import Image

# from models.inception_resnet_v1 import InceptionResnetV1
from torchvision.models.inception import inception_v3 , Inception_V3_Weights

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
import mediapipe as mp


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
vidcap = cv2.VideoCapture(-1)

old = [0,0,0,0]
count = 0
myfolder = os.path.dirname(__file__) 

def load_data():
    data_transform = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(myfolder + '/data', x),
                                            data_transform[x])
                    for x in ['train', 'valid']}
    classes = image_datasets['valid'].classes
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=40,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'valid']}
    return dataloaders, classes
    

def capture_dataset():
    count = 0
    while count < 200:
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
            # cv2.putText(frame, 'test', (x+w/2, y), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2, cv2.LINE_AA, True)
            cv2.putText(frame, "a string", (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            cv2.imshow('frame', frame)
            if count < 150:
                cv2.imwrite(myfolder +'/data/train/thomas/thomas'+ str(count)+ '.png', resized)
            else:
                cv2.imwrite(myfolder +'/data/valid/thomas/thomas'+ str(count)+ '.png', resized)
            count += 1

        if len(face) != 0:
            old = face[0]
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vidcap.release()
    cv2.destroyAllWindows()


# def live_stream():
#     while True:
#         ret, frame = vidcap.read()
#         frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#         face = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
#         if len(face) == 0:
#             print("No face detected")
#         else:
#             x, y , w, h = face[0]
#             cv2.rectangle(frame,(x,y), (x+w,y+h),(0,255,0),2)
#             cv2.imshow('frame', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     vidcap.release()
#     cv2.destroyAllWindows()

def plot_history(train_losses, train_accuracies, valid_losses, valid_accuracies):
    plt.figure(figsize=(7, 3))

    plt.subplot(1, 2, 1)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    p = plt.plot(train_losses, label='train')
    plt.plot(valid_losses, label='valid')
    plt.ylim(0, 2)
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    p = plt.plot(train_accuracies, label='train')
    plt.plot(valid_accuracies, label='valid')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

def one_epoch(model, data_loader, opt=None):
    device = next(model.parameters()).device
    train = False if opt is None else True
    model.train() if train else model.eval()
    losses, correct, total = [], 0, 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            logits = model(x)
        loss = F.cross_entropy(logits, y)
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()

        losses.append(loss.item())
        total += len(x)
        correct += (torch.argmax(logits, dim=1) == y).sum().item()
    return np.mean(losses), correct / total

def train(model, loader_train, loader_valid, lr=1e-3, max_epochs=7, weight_decay=1e-1, patience=20):
    train_losses, train_accuracies = [], []
    valid_losses, valid_accuracies = [], []

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_valid_accuracy = 0
    best_valid_accuracy_epoch = 0

    t = tqdm(range(max_epochs))
    for epoch in t:
        train_loss, train_acc = one_epoch(model, loader_train, opt)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        valid_loss, valid_acc = one_epoch(model, loader_valid)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_acc)

        t.set_description(f'train_acc: {train_acc:.2f}, valid_acc: {valid_acc:.2f}')

        if valid_acc > best_valid_accuracy:
            best_valid_accuracy = valid_acc
            best_valid_accuracy_epoch = epoch

        if epoch > best_valid_accuracy_epoch + patience:
            break
    t.set_description(f'best valid acc: {best_valid_accuracy:.2f}')
    return train_losses, train_accuracies, valid_losses, valid_accuracies

def live_stream(model, device,labels):
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    while True:
        ret, frame = vidcap.read()
        frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
        if len(face) == 0:
            print("No face detected")
        else:
            x, y , w, h = face[0]
            roi_color = frame[y:y+h,x:x+w]
            cv2.rectangle(frame,(x,y), (x+w,y+h),(0,255,0),2)
            im_pil = Image.fromarray(roi_color)
            input = preprocess(im_pil)
            input = input.unsqueeze(0)
            input = input.to(device)
            with torch.no_grad():
                output = model(input)
            output_softmax = torch.nn.functional.softmax(output,dim=1)
            output_index = torch.argmax(output_softmax,dim=1)
            label = str(labels[output_index])
            frame = cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
            cv2.imshow("Window", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vidcap.release()
    cv2.destroyAllWindows()

class FaceDetector():
    def __init__(self, minDetectionCon=0.6):
 
        self.minDetectionCon = minDetectionCon
 
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)
 
    def findFaces(self, img, draw=True):
 
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        # print(self.results)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    img = self.fancyDraw(img,bbox)
 
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                            (bbox[0] + 10, bbox[1] + bbox[3]+20), cv2.FONT_HERSHEY_PLAIN,
                            1, (0, 255, 0), 2)
        return img, bboxs
 
    def fancyDraw(self, img, bbox, l=30, t=5, rt= 1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
 
        cv2.rectangle(img, bbox, (0, 255, 0), rt)
        # Top Left  x,y
        cv2.line(img, (x, y), (x + l, y), (0, 0, 255), t)
        cv2.line(img, (x, y), (x, y+l), (0, 0, 255), t)
        # Top Right  x1,y
        cv2.line(img, (x1, y), (x1 - l, y), (0, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y+l), (0, 0, 255), t)
        # Bottom Left  x,y1
        cv2.line(img, (x, y1), (x + l, y1), (0, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (0, 0, 255), t)
        # Bottom Right  x1,y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (0, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (0, 0, 255), t)
        return img
 

def media_pipe(model, device,labels, predict = False):
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    pTime = 0
    count = 2413
    detector = FaceDetector()
    while True:
        success, img_ori = vidcap.read()
        img, bboxs = detector.findFaces(img_ori.copy())
        if(len(bboxs) == 0):
            print("No faces found")
        else:
            for box in bboxs:
                x,y,w,h = box[1]
                roi_color = img_ori[y:y+h,x:x+w]
                # cv2.rectangle(frame,(x,y), (x+w,y+h),(0,255,0),2)
                # print(count)
                # if count < 3000:
                #     cv2.imwrite(myfolder +'/data/train/thomas/thomas'+ str(count)+ '.png', roi_color)
                # else:
                #     cv2.imwrite(myfolder +'/data/valid/thomas/thomas'+ str(count)+ '.png', roi_color)
                # if count == 4000:
                #     break
                # count += 1
                if predict:
                    im_pil = Image.fromarray(roi_color)
                    input = preprocess(im_pil)
                    input = input.unsqueeze(0)
                    input = input.to(device)
                    with torch.no_grad():
                        output = model(input)
                    output_softmax = torch.nn.functional.softmax(output,dim=1)
                    output_index = torch.argmax(output_softmax,dim=1)
                    label = str(labels[output_index])
                    img = cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
            # cv2.imshow("Window", frame)
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, f'FPS: {int(fps)}', (10, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vidcap.release()
    cv2.destroyAllWindows()


def train_net():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloader, labels = load_data()
    pretrained_model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    pretrained_model.fc = nn.Linear(pretrained_model.fc.in_features, 3)
    pretrained_model.aux_logits=False
    pretrained_model = pretrained_model.to(device)
    plot_history(*train(pretrained_model,dataloader['train'], dataloader['valid']))
    media_pipe(pretrained_model,device, labels, True)

    cv2.waitKey(0)




train_net()
# capture_dataset()
# live_stream()
# media_pipe(1,2,3)

