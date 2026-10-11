import sys
import cv2
import numpy
import scipy
import mediapipe as mp
import pyqtgraph
from PySide6 import QtCore

print("python   :", sys.version.split()[0])
print("numpy    :", numpy.__version__)
print("opencv   :", cv2.__version__)
print("mediapipe:", mp.__version__, "| punya mp.solutions:", hasattr(mp, "solutions"))
print("PySide6  :", QtCore.__version__)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
ok, frame = cap.read()
print("kamera   :", ok, frame.shape if ok else None)
if ok:
    print("fps dilaporkan:", cap.get(cv2.CAP_PROP_FPS))
cap.release()