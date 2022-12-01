
import cv2
import face_detection
import math
import pdb
import numpy as np

print(face_detection.available_detectors)
detector = face_detection.build_detector(
  "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)

import os
imgs = list(os.listdir())
imgs = [ x for x in imgs if '.jpg' in x ]
imgs = [ x for x in imgs if '_graycrop.jpg' not in x ]
imgs.sort()

for img_name in imgs:
  print(img_name + " crop & gray ed")
  img = cv2.imread(img_name)[:, :, ::-1]
  h,w,c = img.shape
  detections = detector.detect(img)
  if len(detections) > 0:
    x1, y1, x2, y2, _ = detections[0]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    sx, ex = sorted([x1, x2])
    sy, ey = sorted([y1, y2])
  else:
    cv2.imshow("img", img)
    sx, sy = input("sx,sy=").strip().split(',')
    ex, ey = input("ex,ey=").strip().split(',')
    sx, sy, ex, ey = int(sx), int(sy), int(ex), int(ey)
  cx, cy = (sx+ex)/2, (sy+ey)/2
  dw = ex - sx
  dh = (dw*3 - (ey - sy))/2
  rsx = 0 if sx - dw < 0 else int(sx - dw)
  rex = w-1 if ex + dw >= w else int(ex + dw)
  rsy = 0 if sy - dh < 0 else int(sy - dh)
  rey = h-1 if ey + dh >= h else int(ey + dh)
  roiW = rex - rsx
  roiH = rey - rsy
  #pdb.set_trace()
  if roiW > roiH:
    dd = roiW - roiH
    d = dd/2
    rsx = int(rsx + d)
    rex = int(rex - d)
  elif roiW <= roiH:
    dd = roiH - roiW
    d = dd/2
    rsy = int(rsy + d)
    rey = int(rey - d)
  img = img.copy()
  print("roi(rsx,rsy,rex,rey): %d,%d,%d,%d"%(rsx,rsy,rex,rey))
  crop = img[rsy:rey, rsx:rex].copy()
  cv2.rectangle(img, (int(sx), int(sy)), (int(ex),int(ey)), (0,0,255),1)
  cv2.rectangle(img, (rsx, rsy), (rex, rey), (125,125,0),1)
  resize = cv2.resize(crop, (320, 320))
  gray = cv2.cvtColor(resize, cv2.COLOR_RGB2GRAY)

  cv2.imshow(img_name, img)
  cv2.imshow(img_name + " gray", gray)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  out_name = ".".join(img_name.split('.')[:-1]) + "_graycrop.jpg"
  cv2.imwrite(out_name, gray)

  #pdb.set_trace()
