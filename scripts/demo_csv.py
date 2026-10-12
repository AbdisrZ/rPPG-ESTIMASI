"""Putar ulang results/<video>_rgb.csv lewat StreamingProcessor (jalur sama dengan live)."""
import argparse
import pandas as pd
from rppg.pipeline import StreamingProcessor

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--keluar", default=None)
a = ap.parse_args()

df = pd.read_csv(a.csv)
df = df[df.roi_ada == 1] if "roi_ada" in df else df
proc = StreamingProcessor()
baris = []
for r in df.itertuples():
    k = proc.push_rgb([r.r, r.g, r.b], r.t)
    if k.bpm is not None:
        baris.append((k.t, k.bpm, k.confidence))
        print(f"t={k.t:6.1f}s  BPM={k.bpm:6.1f}  conf={k.confidence:.2f}")
if a.keluar:
    pd.DataFrame(baris, columns=["t", "bpm", "confidence"]).to_csv(a.keluar, index=False)
