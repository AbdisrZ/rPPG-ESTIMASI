import time
import numpy as np
import cv2


def ukur(nama, setup, detik=5):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    setup(cap)
    for _ in range(10):
        cap.read()
    ts = []
    t_end = time.perf_counter() + detik
    while time.perf_counter() < t_end:
        ok, _ = cap.read()
        if not ok:
            break
        ts.append(time.perf_counter())
    cap.release()
    dt = np.diff(ts)
    print(f"{nama:22s} fps={1 / dt.mean():5.1f}  jitter={dt.std() * 1000:5.1f} ms  frame={len(ts)}")


def default(cap):
    pass


def mjpg(cap):
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)


def yuy2(cap):
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUY2"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)


ukur("default", default)
ukur("MJPG 640x480 30fps", mjpg)
ukur("YUY2 640x480 30fps", yuy2)