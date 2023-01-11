import cv2
import mediapipe as mp
import time
 
 
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
                            (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
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
 
 
def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    count = 0
    detector = FaceDetector()
    while True:
        success, img_ori = cap.read()
        img, bboxs = detector.findFaces(img_ori.copy())
        if(len(bboxs) == 0):
            print("No faces found")
        else:
            x,y,w,h = bboxs[0][1]
            print(x)
            face_size = img_ori[y:y+h,x:x+w]
            face_img = cv2.resize(face_size, (224,224), interpolation = cv2.INTER_AREA)

        #print(bboxs)
 
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.imshow("Image", img)
        cv2.imshow("Face Img", face_img)
        #cv2.imwrite('C:/Users/Thoma/OneDrive/Desktop/UNI/E22/Deep Neural Networks/dnn/DeepLearningProject/ournet/face_thomas/thomas'+ str(count)+ '.png', resized)
        count += 1
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
 
# Blot en test her
main()

if __name__ == "_main_":
    main()