import time
import numpy as np
import cv2

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
for _ in range(10):
    cap.read()

ts, unik = [], []
prev = None
t_end = time.perf_counter() + 10
while time.perf_counter() < t_end:
    ok, f = cap.read()
    if not ok:
        break
    ts.append(time.perf_counter())
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    unik.append(prev is None or not np.array_equal(g, prev))
    prev = g
cap.release()

total = ts[-1] - ts[0]
dt = np.diff(ts) * 1000
print(f"frame dibaca : {len(ts) / total:.1f} per detik")
print(f"frame unik   : {sum(unik) / total:.1f} per detik")
print(f"dt p5/p50/p95: {np.percentile(dt, 5):.1f} / {np.percentile(dt, 50):.1f} / {np.percentile(dt, 95):.1f} ms")
print(f"dt di bawah 5 ms: {(dt < 5).mean() * 100:.0f}%")