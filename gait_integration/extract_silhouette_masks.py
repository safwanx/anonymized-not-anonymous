#!/usr/bin/env python3
"""High-quality person silhouettes via YOLO segmentation, for the gait adversary.

The default Otsu silhouettes in gait_silhouette.py are the weak link in gait
accuracy. This produces real segmentation masks (Ultralytics YOLO-seg), the
clean upgrade to make gait numbers authoritative.

Critical design point: masks are extracted PER METHOD, from that method's
(possibly anonymized) frames, and written to <mask_root>/<method>/<stem>/<frame>.png.
This is deliberate: if body_pixel destroys the silhouette so the segmenter can no
longer find a person, that coverage loss is a real privacy effect we must
measure, not bypass by reusing clean masks. Gallery is always the original
method; probes use each anonymization method.

Output matches gait_silhouette's --mask-dir contract: a binary PNG per frame
(person=255), cropped to the mask's bounding box. cut_img normalizes at eval time.

GPU recommended. Resumable (skips existing masks).

Run on SLURM:
    python gait_integration/extract_silhouette_masks.py \
        --methods original,face_blur,body_blur,body_pixel,deepprivacy2 \
        --out outputs/gait/masks
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm

import gait_paths as G
from pipeline_common import (  # noqa: E402
    DEVICE,
    EVAL_PROTOCOLS,
    available_anonymization_methods,
    frame_root_for_method,
    meta_by_filename,
)

DEFAULT_SEG_WEIGHTS = "yolo11x-seg.pt"  # auto-downloads if absent
PERSON_CLASS = 0  # COCO person


def resolve_weights(name: str) -> str:
    if Path(name).exists():
        return name
    for root in (G.PROJECT_ROOT, G.MODELS):
        cand = root / name
        if cand.exists():
            return str(cand)
    return name  # let ultralytics fetch it


def files_for_method(method: str) -> list[str]:
    """Original needs galleries + probes; anon methods need only their probes."""
    galleries, probes = set(), set()
    for gallery_files, probe_files in EVAL_PROTOCOLS.values():
        galleries.update(gallery_files)
        probes.update(probe_files)
    wanted = (galleries | probes) if method == "original" else probes
    return sorted(f for f in wanted if f in meta_by_filename)


def best_person_mask(result, h: int, w: int) -> np.ndarray | None:
    if result.masks is None or result.boxes is None:
        return None
    cls = result.boxes.cls.cpu().numpy().astype(int)
    conf = result.boxes.conf.cpu().numpy()
    persons = np.where(cls == PERSON_CLASS)[0]
    if persons.size == 0:
        return None
    best = persons[int(np.argmax(conf[persons]))]
    mask = result.masks.data[best].cpu().numpy()  # [mh, mw] in [0,1]
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return (mask > 0.5).astype(np.uint8) * 255


def crop_to_mask(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    return mask[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]


def extract_for_method(model, method: str, out_root: Path, overwrite: bool) -> None:
    files = files_for_method(method)
    frame_root = frame_root_for_method(method)
    print(f"[{method}] {len(files)} videos  frames<-{frame_root}")
    for filename in tqdm(files, desc=f"masks {method}", leave=False):
        stem = str(meta_by_filename[filename]["stem"])
        frames_dir = frame_root / stem
        if not frames_dir.exists():
            continue
        dst_dir = out_root / method / stem
        dst_dir.mkdir(parents=True, exist_ok=True)
        for frame_path in sorted(frames_dir.glob("*.jpg")):
            dst = dst_dir / f"{frame_path.stem}.png"
            if dst.exists() and not overwrite:
                continue
            img = cv2.imread(str(frame_path))
            if img is None:
                continue
            res = model.predict(img, classes=[PERSON_CLASS], verbose=False,
                                device=0 if DEVICE == "cuda" else "cpu")[0]
            mask = best_person_mask(res, img.shape[0], img.shape[1])
            if mask is None:
                continue
            cropped = crop_to_mask(mask)
            if cropped is not None:
                cv2.imwrite(str(dst), cropped)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--methods", default="auto")
    ap.add_argument("--weights", default=DEFAULT_SEG_WEIGHTS)
    ap.add_argument("--out", default=str(G.GAIT_WORK / "masks"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(resolve_weights(args.weights))

    if args.methods.strip().lower() == "auto":
        methods = ["original"] + available_anonymization_methods()
    else:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    out_root = Path(args.out)
    print(f"YOLO-seg masks -> {out_root}  methods={methods}")
    for method in methods:
        extract_for_method(model, method, out_root, args.overwrite)
    print("Done. Feed this dir to eval_gait_identity.py via --mask-root.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
