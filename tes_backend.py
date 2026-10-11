import time
import numpy as np
import cv2


def ukur(nama, backend, w, h, detik=5):
    cap = cv2.VideoCapture(0, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, 30)
    for _ in range(10):
        cap.read()
    ts = []
    t_end = time.perf_counter() + detik
    while time.perf_counter() < t_end:
        ok, _ = cap.read()
        if not ok:
            break
        ts.append(time.perf_counter())
    dapat = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    if len(ts) < 2:
        print(f"{nama:6s} diminta {w}x{h}: gagal membuka atau membaca")
        return
    dt = np.diff(ts)
    print(f"{nama:6s} diminta {w}x{h} dapat {dapat[0]}x{dapat[1]}  "
          f"fps={1 / dt.mean():5.1f}  jitter={dt.std() * 1000:5.1f} ms")


for nama, be in [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)]:
    for w, h in [(640, 480), (1280, 720)]:
        ukur(nama, be, w, h)