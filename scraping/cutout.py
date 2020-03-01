import numpy as np
import cv2
import sys
import os

drawing = False
sx, sy = 0, 0 
gx, gy = 0, 0
rectangles = []
ok = False
directory_name = "kasago"

def draw_circle(event,x,y,flags,param):
    global sx, sy, gx, gy, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        sx,sy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if x > 0 and x < img.shape[1]:
            gx = x
        if y > 0 and y < img.shape[0]:
            gy = y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        w = abs(sx-x)
        h = abs(sy-y)
        if w < h * 1.3 and h < w * 1.3:
            rectangles.append([(sx, sy), (x, y)])

img = np.zeros((256, 256, 3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

files = os.listdir(os.path.join('./image_source/',directory_name))
if not os.path.exists(os.path.join('./image_cutout', directory_name)):
    os.mkdir(os.path.join('./image_cutout', directory_name))

i = 0
while i < len(files):
    rectangles = []
    while True:
        img = cv2.imread(os.path.join('./image_source/',directory_name, files[i]))
        for r in rectangles:
            cv2.rectangle(img, r[0], r[1], (255,255,255), 2)
        if drawing:
            w = abs(sx-gx)
            h = abs(sy-gy)
            if w < h*1.3 and h < w*1.3:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            cv2.rectangle(img, (sx,sy), (gx,gy), color, 2)
        cv2.imshow('image', img)
        
        k = cv2.waitKey(1) & 0xFF
        if k == ord('d'):
            if rectangles:
                rectangles.pop()
        elif k == ord('n'):
            if len(rectangles) > 0:
                for r in rectangles:
                    x = min(r[0][0], r[1][0])
                    y = min(r[0][1], r[1][1])
                    w = abs(r[0][0] - r[1][0])
                    h = abs(r[0][1] - r[1][1])
                    
                    cutout = img[y:y+h,x:x+w]
                    cv2.imwrite(os.path.join('./image_cutout/',directory_name,files[i]), cutout)
            i += 1
            break
        elif k == ord('b'):
            if i > 1:
                i -= 2
                break
        elif k == ord('q'):
            sys.exit()