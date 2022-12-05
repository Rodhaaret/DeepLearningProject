import cv2
import os

face_cascades = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_faces(img, draw_box=True):
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascades.detectMultiScale(grayscale_img,scaleFactor=1.6)

    for (x,y,width,height) in faces:
        if draw_box:
            cv2.rectangle(img,(x,y),(x+width,y+height),(0,255,0),5)
        
        face_box =img[y:y+height,x:x+height]
        face_coords = [x,y,width,height]

    return img, face_box, face_coords


if __name__ == "__main__":
	files = os.listdir('sample_faces')
	images = [file for file in files if 'jpg' in file]
	for image in images:
		img = cv2.imread('sample_faces/' + image)
		detected_faces, _, _ = detect_faces(img)
		cv2.imwrite('sample_faces/detected_faces/' + image, detected_faces)