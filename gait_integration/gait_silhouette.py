"""Silhouette sequence extraction for the gait adversary.

GaitBase consumes an ordered sequence of normalized binary silhouettes (64x44),
not RGB. We reuse the detection boxes already computed by the pipeline
(detections/<stem>.json), crop the person, binarize, and apply OpenGait's
canonical cut_img normalization so the input distribution matches training.

Silhouette source:
  - default: Otsu threshold on the grayscale person crop (no extra model needed,
    works on NTU's relatively clean backgrounds; same family as the existing
    silhouette proxy but kept as an ordered sequence).
  - optional: precomputed masks via --mask-dir (drop-in for a real segmentation
    model later); expects <mask_dir>/<stem>/<frame>.png binary masks.

cut_img is the standard OpenGait pretreatment (datasets/pretreatment.py):
crop to the silhouette's vertical extent, scale to height 64, center on the
horizontal mass median, crop/pad width to 44.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import gait_paths as G
from pipeline_common import DETECT_DIR, clip_box, read_json  # noqa: E402


def cut_img(img: np.ndarray, t_h: int = G.SIL_H, t_w: int = G.SIL_W) -> np.ndarray | None:
    """Normalize a binary silhouette (uint8 0/255) to t_h x t_w, OpenGait-style."""
    if img.sum() <= 10000:  # too few foreground pixels to be a usable silhouette
        return None
    y = img.sum(axis=1)
    y_top = int((y != 0).argmax())
    y_btm = int((y != 0).cumsum().argmax())
    img = img[y_top:y_btm + 1, :]
    if img.shape[0] == 0:
        return None
    ratio = img.shape[1] / img.shape[0]
    img = cv2.resize(img, (int(t_h * ratio), t_h), interpolation=cv2.INTER_CUBIC)

    total = img.sum()
    col_cumsum = img.sum(axis=0).cumsum()
    x_center = int(np.searchsorted(col_cumsum, total / 2))
    half = t_w // 2
    left, right = x_center - half, x_center + half
    if left < 0 or right > img.shape[1]:
        pad = np.zeros((t_h, half), dtype=img.dtype)
        img = np.concatenate([pad, img, pad], axis=1)
        left, right = left + half, right + half
    img = img[:, left:right]
    if img.shape[1] != t_w:  # guard against off-by-one
        img = cv2.resize(img, (t_w, t_h), interpolation=cv2.INTER_CUBIC)
    return img.astype(np.uint8)


def _silhouette_from_crop(gray_crop: np.ndarray) -> np.ndarray:
    _thr, binary = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def extract_sequence(
    stem: str,
    frame_root: Path,
    frame_step: int = 1,
    mask_dir: Path | None = None,
    min_frames: int = 10,
) -> np.ndarray | None:
    """Return [S, 64, 44] float in [0,1], or None if too few valid silhouettes.

    Boxes come from the original detections (detections/<stem>.json); the image
    pixels come from frame_root (original or an anonymized method dir), matching
    the existing pixel-space attackers.
    """
    frames_dir = frame_root / stem
    if not frames_dir.exists():
        return None
    detections = read_json(DETECT_DIR / f"{stem}.json", {})

    sils = []
    for frame_path in sorted(frames_dir.glob("*.jpg"))[::frame_step]:
        if mask_dir is not None:
            mpath = mask_dir / stem / f"{frame_path.stem}.png"
            if not mpath.exists():
                continue
            binary = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
            if binary is None:
                continue
            _t, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
        else:
            det = detections.get(frame_path.stem, {})
            persons = det.get("persons", [])
            if not persons:
                continue
            image = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            best = max(persons, key=lambda b: b[4])
            clipped = clip_box(best, image.shape[1], image.shape[0])
            if clipped is None:
                continue
            x1, y1, x2, y2 = clipped
            binary = _silhouette_from_crop(image[y1:y2, x1:x2])

        norm = cut_img(binary)
        if norm is not None:
            sils.append(norm.astype(np.float32) / 255.0)

    if len(sils) < min_frames:
        return None
    return np.stack(sils)
