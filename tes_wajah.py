import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
       "blaze_face_short_range/float16/1/blaze_face_short_range.tflite")
MODEL = "blaze_face_short_range.tflite"
if not os.path.exists(MODEL):
    urllib.request.urlretrieve(URL, MODEL)

opts = vision.FaceDetectorOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL),
    running_mode=vision.RunningMode.IMAGE,
)
detector = vision.FaceDetector.create_from_options(opts)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
n, t0 = 0, time.perf_counter()
while True:
    ok, frame = cap.read()
    if not ok:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    for d in res.detections:
        b = d.bounding_box
        cv2.rectangle(frame, (b.origin_x, b.origin_y),
                      (b.origin_x + b.width, b.origin_y + b.height), (0, 255, 0), 2)
    n += 1
    fps = n / (time.perf_counter() - t0)
    cv2.putText(frame, f"{fps:.1f} fps | wajah: {len(res.detections)}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("tes deteksi wajah", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()