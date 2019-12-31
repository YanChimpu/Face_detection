import cv2
import dlib
import argparse
import time
from imutils import face_utils
import imutils

cap = cv2.VideoCapture("/Users/pu/Documents/work/data/temp/3-NB3/算法-正常-郭新宇.mp4")
cnn_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = face_utils.FaceAligner(predictor, desiredFaceWidth=256)
font = cv2.FONT_HERSHEY_SIMPLEX


def rect_to_bb(rect): # 获得人脸矩形的坐标信息
    x,y,w,h = 0,0,0,0
    for i in rect:
        x = i.left()
        y = i.top()
        w = i.right() - x
        h = i.bottom() - y
    return (x, y, w, h)

def cnn_rect_to_bb(rect):
    x,y,w,h = 0,0,0,0
    for face in rect:
        x = face.rect.left()
        y = face.rect.top()
        w = face.rect.right() - x
        h = face.rect.bottom() - y
    return (x,y,w,h)

filename = 0
start = time.time()
while(True):
    ret, frame = cap.read()
    rects1,rects2 = None,None
    img_height, img_width = image.shape[:2]
    if ret:
        #frame = cv2.resize(frame,(200,354))
        rects1 = detector(frame,0)
        if len(rects1) > 0:
            (x1, y1, w1, h1) = rect_to_bb(rects1)
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
        rects2 = cnn_face_detector(frame,0)
        if len(rects2) > 0:
            (x2, y2, w2, h2) = cnn_rect_to_bb(rects2)
            image = cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 0), 2)
        cv2.putText(image, "HOG", (img_width-50,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0,255,0), 2)
        cv2.putText(image, "CNN", (img_width-50,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255,0,0), 2)
        cv2.imwrite('res_img/'+str(filename)+".jpg",frame)
        filename += 1 
    else:
        break
end = time.time()
total_time = format(end-start,'.2f')
print("Total time used:",total_time,"s")
print("Per image time used:",float(total_time)/filename,"s")
cap.release()
cv2.destroyAllWindows()
