#!/usr/bin/env python3
# Shared configuration and helpers for the split ACCV pilot pipeline.
# Importing this module validates metadata and split files, but does not run
# extraction, detection, anonymization, or evaluation stages.

# 0.2 - Imports
from __future__ import annotations

import csv
import gc
import json
import math
import os
import re
import shutil
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# %%
# 0.3 - Path configuration


def _notebook_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else None


def _default_project_root() -> Path:
    env_root = _env_path("ACCV_ROOT")
    if env_root is not None:
        return env_root

    local_dir = _notebook_dir()
    cwd = Path.cwd().resolve()
    candidates = [local_dir.parent, local_dir, cwd, *cwd.parents]
    for candidate in candidates:
        if (
            (candidate / "protocol" / "metadata.csv").exists()
            or (candidate / "pilot" / "metadata.csv").exists()
            or (candidate / "metadata.csv").exists()
        ):
            return candidate.resolve()
    for candidate in candidates:
        if (candidate / "repos").exists() or (candidate / "models").exists():
            return candidate.resolve()
    return local_dir


PROJECT_ROOT = _default_project_root()


def _default_pilot_root(project_root: Path) -> Path:
    env_root = _env_path("PILOT_ROOT")
    if env_root is not None:
        return env_root

    candidates = [
        project_root / "protocol",
        project_root / "pilot",
        project_root,
    ]
    for candidate in candidates:
        if (candidate / "metadata.csv").exists():
            return candidate.resolve()
    return (project_root / "protocol").resolve()


ROOT = _default_pilot_root(PROJECT_ROOT)
CACHE_ROOT = (_env_path("ACCV_CACHE") or Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "accv").resolve()
REPOS = (_env_path("REPOS_DIR") or CACHE_ROOT / "repos").resolve()
MODELS = (_env_path("MODELS_DIR") or CACHE_ROOT / "models").resolve()
NTU_ROOT = (_env_path("NTU_ROOT") or PROJECT_ROOT).resolve()
OUTPUT_ROOT = (_env_path("PILOT_OUTPUT_ROOT") or PROJECT_ROOT / "outputs").resolve()

FRAMES_DIR = OUTPUT_ROOT / "frames"
DETECT_DIR = OUTPUT_ROOT / "detections"
ANON_DIR = OUTPUT_ROOT / "anonymized"
FEATURES_DIR = OUTPUT_ROOT / "features"
LOGS_DIR = OUTPUT_ROOT / "logs"
SPLITS_DIR = ROOT / "splits"
META_CSV = ROOT / "metadata.csv"

for directory in [FRAMES_DIR, DETECT_DIR, ANON_DIR, FEATURES_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

print(f"PROJECT_ROOT      : {PROJECT_ROOT}")
print(f"PILOT_ROOT        : {ROOT}")
print(f"PILOT_OUTPUT_ROOT : {OUTPUT_ROOT}")
print(f"REPOS_DIR         : {REPOS}")
print(f"MODELS_DIR        : {MODELS}")
print(f"NTU_ROOT          : {NTU_ROOT}")
print(f"metadata          : {META_CSV} (exists={META_CSV.exists()})")
print(f"splits            : {SPLITS_DIR} (exists={SPLITS_DIR.exists()})")

# %%
# 0.4 - Runtime configuration
FRAME_STRIDE = int(os.environ.get("FRAME_STRIDE", "2"))
FRAME_JPEG_QUALITY = int(os.environ.get("FRAME_JPEG_QUALITY", "85"))
ANON_JPEG_QUALITY = int(os.environ.get("ANON_JPEG_QUALITY", "85"))
DETECTION_BATCH_SIZE = int(os.environ.get("DETECTION_BATCH_SIZE", "16"))
FACE_FRAME_STEP = int(os.environ.get("FACE_FRAME_STEP", "3"))
REID_FRAME_STEP = int(os.environ.get("REID_FRAME_STEP", "3"))
SIL_FRAME_STEP = int(os.environ.get("SIL_FRAME_STEP", "1"))
SIL_MAX_EVAL = int(os.environ.get("SIL_MAX_EVAL", "800"))  # 0 means all
UTILITY_MAX_FRAMES = int(os.environ.get("UTILITY_MAX_FRAMES", "12"))
UTILITY_FRAME_SIZE = int(os.environ.get("UTILITY_FRAME_SIZE", "32"))
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))
ALLOW_INCOMPLETE_PILOT = os.environ.get("ALLOW_INCOMPLETE_PILOT", "0").lower() in {"1", "true", "yes"}

YOLO_MODEL = os.environ.get("YOLO_MODEL", "yolov8x.pt")

print(f"FRAME_STRIDE      : {FRAME_STRIDE}")
print(f"DETECTION_BATCH   : {DETECTION_BATCH_SIZE}")
print(f"SIL_MAX_EVAL      : {SIL_MAX_EVAL if SIL_MAX_EVAL else 'all'}")
print(f"ALLOW_INCOMPLETE  : {ALLOW_INCOMPLETE_PILOT}")

# %%
# 0.5 - GPU check and common helpers
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    print(
        f"GPU: {torch.cuda.get_device_name(0)}, "
        f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    )
else:
    print("WARNING: no CUDA GPU detected. Heavy sections will be slow.")


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def stem_of(filename: str) -> str:
    return filename.replace("_rgb.avi", "")


def parse_filename(name: str) -> dict[str, int] | None:
    match = re.match(r"S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})", name)
    if not match:
        return None
    return {
        key: int(value)
        for key, value in zip(
            ["setup", "camera", "person", "replication", "action"],
            match.groups(),
        )
    }


def looks_absolute_path(raw_path: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", raw_path)) or raw_path.startswith("/")


def _existing_path(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _candidate_data_roots() -> list[Path]:
    roots = [
        ROOT,
        ROOT.parent,
        NTU_ROOT,
        PROJECT_ROOT,
        PROJECT_ROOT / "datasets",
        PROJECT_ROOT / "datasets" / "ntu_rgbd120",
        PROJECT_ROOT / "datasets" / "NTU_RGBD120",
    ]
    unique: list[Path] = []
    seen = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            unique.append(root)
            seen.add(key)
    return unique


def resolve_rgb_path(row: dict[str, Any]) -> Path:
    raw = str(row.get("rgb_path", "")).strip()
    filename = str(row["filename"])
    setup = int(row["setup"])
    setup_dir = f"nturgbd_rgb_s{setup:03d}"
    candidates: list[Path] = []

    if raw:
        raw_norm = raw.replace("\\", "/")
        raw_path = Path(raw)
        candidates.append(raw_path)
        if not looks_absolute_path(raw):
            candidates.append(ROOT / raw_norm)

    for base in _candidate_data_roots():
        candidates.extend(
            [
                base / "rgb" / filename,
                base / setup_dir / filename,
                base / "datasets" / setup_dir / filename,
                base / "ntu_rgbd120" / setup_dir / filename,
            ]
        )

    found = _existing_path(candidates)
    if found is None:
        raise FileNotFoundError(
            f"Could not resolve RGB video for {filename}. "
            f"Set PILOT_ROOT for a packed pilot or NTU_ROOT for full NTU folders."
        )
    return found


def resolve_skeleton_path(row: dict[str, Any]) -> Path | None:
    if str(row.get("has_skeleton", "")).lower() != "true":
        return None

    raw = str(row.get("skeleton_path", "")).strip()
    filename = str(row["filename"])
    setup = int(row["setup"])
    skeleton_name = filename.replace("_rgb.avi", ".skeleton")
    candidates: list[Path] = []

    if raw:
        raw_norm = raw.replace("\\", "/")
        raw_path = Path(raw)
        candidates.append(raw_path)
        if not looks_absolute_path(raw):
            candidates.append(ROOT / raw_norm)

    skeleton_archive = (
        Path("nturgbd_skeletons_s001_to_s017") / "nturgb+d_skeletons"
        if setup <= 17
        else Path("nturgbd_skeletons_s018_to_s032")
    )
    for base in _candidate_data_roots():
        candidates.extend(
            [
                base / "skeletons" / skeleton_name,
                base / skeleton_archive / skeleton_name,
                base / "datasets" / skeleton_archive / skeleton_name,
                base / "ntu_rgbd120" / skeleton_archive / skeleton_name,
            ]
        )

    return _existing_path(candidates)


def safe_limit(files: list[str], max_items: int, seed: int = RANDOM_SEED) -> list[str]:
    if max_items <= 0 or len(files) <= max_items:
        return files
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(files), size=max_items, replace=False).tolist())
    return [files[i] for i in idx]


def load_split(filename: str) -> list[str]:
    path = SPLITS_DIR / filename
    if not path.exists():
        print(f"WARNING: missing split file {path}")
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def count_jpgs(path: Path) -> int:
    return sum(1 for _ in path.glob("*.jpg")) if path.exists() else 0


def sample_frame_paths(frame_dir: Path, max_frames: int) -> list[Path]:
    paths = sorted(frame_dir.glob("*.jpg"))
    if not paths:
        return []
    if len(paths) <= max_frames:
        return paths
    indices = np.linspace(0, len(paths) - 1, num=max_frames, dtype=int)
    return [paths[int(i)] for i in indices]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.symlink(src, dst)
        return
    except OSError:
        pass
    try:
        os.link(src, dst)
        return
    except OSError:
        shutil.copy2(src, dst)


def ensure_insightface_root(model_name: str = "buffalo_l") -> str:
    """Return a root path where insightface can find models/<model_name>/*.onnx."""
    env_root = _env_path("INSIGHTFACE_ROOT")
    if env_root and (env_root / "models" / model_name).exists():
        return str(env_root)

    packs_root = MODELS / "face_arcface" / "insightface_model_packs"
    source_candidates = [
        packs_root / "models" / model_name,
        packs_root / model_name / model_name,
        packs_root / model_name,
    ]
    source = _existing_path(source_candidates)
    if source is None or not list(source.glob("*.onnx")):
        raise FileNotFoundError(
            f"Could not find InsightFace ONNX pack for {model_name} under {packs_root}"
        )

    runtime_root = (
        _env_path("INSIGHTFACE_RUNTIME_ROOT")
        or OUTPUT_ROOT / "model_runtime" / "insightface"
    )
    target = runtime_root / "models" / model_name
    target.mkdir(parents=True, exist_ok=True)
    for onnx_file in source.glob("*.onnx"):
        link_or_copy(onnx_file, target / onnx_file.name)
    return str(runtime_root)


# %% [markdown]

# 1.1 - Load and validate metadata
def load_metadata() -> list[dict[str, Any]]:
    if not META_CSV.exists():
        raise FileNotFoundError(f"Missing metadata.csv: {META_CSV}")

    with META_CSV.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        for key in ["setup", "camera", "person", "replication", "action"]:
            row[key] = int(row[key])
        row["stem"] = stem_of(row["filename"])
        row["rgb_resolved"] = str(resolve_rgb_path(row))
        skeleton_path = resolve_skeleton_path(row)
        row["skeleton_resolved"] = str(skeleton_path) if skeleton_path else ""

    return rows


meta = load_metadata()
meta_by_filename = {row["filename"]: row for row in meta}
meta_by_stem = {row["stem"]: row for row in meta}

persons = sorted({row["person"] for row in meta})
actions = sorted({row["action"] for row in meta})
setups = sorted({row["setup"] for row in meta})
pools = Counter(row["pool"] for row in meta)
skeleton_count = sum(1 for row in meta if row["skeleton_resolved"])

print(f"Total rows        : {len(meta)}")
print(f"Persons           : {len(persons)}")
print(f"Actions           : {len(actions)}")
print(f"Setups            : {setups}")
print(f"Pools             : {dict(pools)}")
print(f"Resolved skeletons: {skeleton_count}/{len(meta)}")

if len(actions) != 120:
    raise RuntimeError(f"Expected all 120 NTU actions, found {len(actions)}")
if skeleton_count != len(meta):
    message = f"Some pilot skeleton paths did not resolve: {skeleton_count}/{len(meta)}"
    if not ALLOW_INCOMPLETE_PILOT:
        raise RuntimeError(message)
    print(f"WARNING: {message}")
    print("WARNING: proceeding without complete skeleton paths for smoke/debug run only.")
if len(meta) != 21600:
    message = (
        f"Expected 21600 pilot videos, found {len(meta)}. "
        "Restore the released protocol/metadata.csv and verify it against protocol/MANIFEST.sha256."
    )
    if not ALLOW_INCOMPLETE_PILOT:
        raise RuntimeError(message)
    print(f"WARNING: {message}")
    print("WARNING: proceeding with incomplete pilot for smoke/debug run only.")

# %%
# 1.2 - Load corrected split files
gallery_xv_files = load_split("gallery_crossview.txt")
probe_xv_c2_files = load_split("probe_crossview_cam2.txt")
probe_xv_c3_files = load_split("probe_crossview_cam3.txt")
gallery_xs_files = load_split("gallery_crosssetup.txt")
probe_xs_files = load_split("probe_crosssetup.txt")
gallery_xr_files = load_split("gallery_crossrange.txt")
probe_xr_files = load_split("probe_crossrange.txt")

EVAL_PROTOCOLS = {
    "crossview_cam2": (gallery_xv_files, probe_xv_c2_files),
    "crossview_cam3": (gallery_xv_files, probe_xv_c3_files),
    "crosssetup": (gallery_xs_files, probe_xs_files),
    "crossrange": (gallery_xr_files, probe_xr_files),
}

# Restrict an experiment to selected protocols without editing each evaluator.
# Leaving the variable unset evaluates all four protocols.
_protocol_filter = os.environ.get("EVAL_PROTOCOL_NAMES", "").strip()
if _protocol_filter:
    _requested_protocols = [name.strip() for name in _protocol_filter.split(",") if name.strip()]
    _unknown_protocols = [name for name in _requested_protocols if name not in EVAL_PROTOCOLS]
    if _unknown_protocols:
        raise RuntimeError(
            f"Unknown EVAL_PROTOCOL_NAMES values: {_unknown_protocols}. "
            f"Known protocols: {sorted(EVAL_PROTOCOLS)}"
        )
    EVAL_PROTOCOLS = {name: EVAL_PROTOCOLS[name] for name in _requested_protocols}
    print(f"Using explicit evaluation protocols from EVAL_PROTOCOL_NAMES: {_requested_protocols}")

for name, (gallery_files, probe_files) in EVAL_PROTOCOLS.items():
    print(f"{name:15s}: gallery={len(gallery_files):5d}, probe={len(probe_files):5d}")
    missing = [filename for filename in gallery_files + probe_files if filename not in meta_by_filename]
    if missing:
        raise RuntimeError(f"{name} has split files missing from metadata, first missing: {missing[0]}")

# %%

# 2.1 - Build video list and completion helpers
all_stems = sorted({row["stem"] for row in meta})
rgb_paths = {row["stem"]: Path(row["rgb_resolved"]) for row in meta}
FRAME_MANIFEST_DIR = FRAMES_DIR / "_manifests"
FRAME_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)


def frame_manifest_path(stem: str) -> Path:
    return FRAME_MANIFEST_DIR / f"{stem}.json"


def video_raw_frame_count(stem: str) -> int | None:
    video_path = rgb_paths.get(stem)
    if video_path is None:
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None
    raw_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return raw_frames if raw_frames > 0 else None


def expected_extracted_frame_count(stem: str) -> int | None:
    raw_frames = video_raw_frame_count(stem)
    if raw_frames is None:
        return None
    return (raw_frames + FRAME_STRIDE - 1) // FRAME_STRIDE


def frame_extraction_complete(stem: str) -> bool:
    manifest = read_json(frame_manifest_path(stem), {})
    out_dir = FRAMES_DIR / stem
    actual = count_jpgs(out_dir)

    if manifest.get("status") == "complete" and int(manifest.get("stride", -1)) == FRAME_STRIDE:
        expected = int(manifest.get("extracted_frames", -1))
        declared_expected = int(manifest.get("expected_extracted_frames", expected))
        if expected > 0 and actual == expected and expected == declared_expected:
            return True

    # Recovery path for already-extracted frame folders with no/stale manifest.
    expected_from_video = expected_extracted_frame_count(stem)
    return expected_from_video is not None and expected_from_video > 0 and actual == expected_from_video


def build_frame_todo() -> list[str]:
    frame_todo = [stem for stem in all_stems if not frame_extraction_complete(stem)]
    print(
        f"Frame extraction: {len(all_stems) - len(frame_todo)} done, "
        f"{len(frame_todo)} to extract"
    )
    return frame_todo

# %%

# 3.2 - Detection completion helpers
def expected_frame_count(stem: str) -> int:
    manifest = read_json(frame_manifest_path(stem), {})
    return int(manifest.get("extracted_frames", count_jpgs(FRAMES_DIR / stem)))


def detection_complete(stem: str) -> bool:
    path = DETECT_DIR / f"{stem}.json"
    if not path.exists():
        return False
    data = read_json(path, {})
    expected = expected_frame_count(stem)
    return expected > 0 and len(data) == expected


def build_detection_todo() -> list[str]:
    det_todo = [
        stem
        for stem in all_stems
        if frame_extraction_complete(stem) and not detection_complete(stem)
    ]
    print(
        f"Detection: {len(all_stems) - len(det_todo)} complete/irrelevant, "
        f"{len(det_todo)} to process"
    )
    return det_todo

# %%

# 4.1 - Anonymization functions
def clip_box(box: list[float], width: int, height: int) -> tuple[int, int, int, int] | None:
    x1 = max(0, int(box[0]))
    y1 = max(0, int(box[1]))
    x2 = min(width, int(box[2]))
    y2 = min(height, int(box[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def apply_blur(image: np.ndarray, boxes: list[list[float]], ksize: int = 51) -> np.ndarray:
    output = image.copy()
    height, width = output.shape[:2]
    kernel = max(ksize, 3) | 1
    for box in boxes:
        clipped = clip_box(box, width, height)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        output[y1:y2, x1:x2] = cv2.GaussianBlur(output[y1:y2, x1:x2], (kernel, kernel), 0)
    return output


def apply_mask(image: np.ndarray, boxes: list[list[float]]) -> np.ndarray:
    output = image.copy()
    height, width = output.shape[:2]
    for box in boxes:
        clipped = clip_box(box, width, height)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        output[y1:y2, x1:x2] = 0
    return output


def apply_pixelate(image: np.ndarray, boxes: list[list[float]], block: int = 12) -> np.ndarray:
    output = image.copy()
    height, width = output.shape[:2]
    for box in boxes:
        clipped = clip_box(box, width, height)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        region = output[y1:y2, x1:x2]
        rh, rw = region.shape[:2]
        small = cv2.resize(
            region,
            (max(1, rw // block), max(1, rh // block)),
            interpolation=cv2.INTER_LINEAR,
        )
        output[y1:y2, x1:x2] = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    return output


METHODS: dict[str, Callable[[np.ndarray, dict[str, Any]], np.ndarray]] = {
    "face_blur": lambda image, det: apply_blur(image, det.get("faces", []), ksize=71),
    "body_blur": lambda image, det: apply_blur(image, det.get("persons", []), ksize=51),
    "body_mask": lambda image, det: apply_mask(image, det.get("persons", [])),
    "body_pixel": lambda image, det: apply_pixelate(image, det.get("persons", []), block=12),
}


def anonymization_complete(method: str, stem: str) -> bool:
    expected = expected_frame_count(stem)
    return expected > 0 and count_jpgs(ANON_DIR / method / stem) == expected


# %%

# 5.0 - Metric helpers
def compute_reid_metrics(
    gallery_feats: np.ndarray,
    gallery_ids: list[int],
    probe_feats: np.ndarray,
    probe_ids: list[int],
) -> dict[str, float]:
    if len(gallery_feats) == 0 or len(probe_feats) == 0:
        return {
            "rank1": 0.0,
            "rank5": 0.0,
            "mAP": 0.0,
            "num_gallery": float(len(gallery_feats)),
            "num_probe": float(len(probe_feats)),
            "queries_with_match": 0.0,
        }

    gallery_norm = gallery_feats / (np.linalg.norm(gallery_feats, axis=1, keepdims=True) + 1e-8)
    probe_norm = probe_feats / (np.linalg.norm(probe_feats, axis=1, keepdims=True) + 1e-8)
    dist = 1 - probe_norm @ gallery_norm.T
    indices = np.argsort(dist, axis=1)

    gallery_ids_np = np.array(gallery_ids)
    probe_ids_np = np.array(probe_ids)

    rank1 = 0
    rank5 = 0
    aps = []
    queries_with_match = 0

    for i, probe_id in enumerate(probe_ids_np):
        sorted_ids = gallery_ids_np[indices[i]]
        matches = sorted_ids == probe_id
        if matches.sum() == 0:
            continue
        queries_with_match += 1
        rank1 += int(bool(matches[0]))
        rank5 += int(bool(matches[:5].any()))
        cumsum = np.cumsum(matches).astype(float)
        precision = cumsum / (np.arange(len(matches)) + 1)
        aps.append(float((precision * matches).sum() / matches.sum()))

    denom = max(len(probe_ids_np), 1)
    return {
        "rank1": rank1 / denom,
        "rank5": rank5 / denom,
        "mAP": float(np.mean(aps)) if aps else 0.0,
        "num_gallery": float(len(gallery_feats)),
        "num_probe": float(len(probe_feats)),
        "queries_with_match": float(queries_with_match),
    }


def frame_root_for_method(method: str) -> Path:
    return FRAMES_DIR if method == "original" else ANON_DIR / method


def files_to_ids(file_list: list[str]) -> list[int]:
    return [int(meta_by_filename[filename]["person"]) for filename in file_list if filename in meta_by_filename]


def save_attack_results(name: str, rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    path = FEATURES_DIR / f"{name}_results.csv"
    df.to_csv(path, index=False)
    print(f"Saved {name} results: {path}")
    if not df.empty:
        display_cols = [
            "protocol",
            "method",
            "rank1",
            "rank5",
            "mAP",
            "probe_coverage",
            "coverage_adjusted_rank1",
        ]
        print(df[[col for col in display_cols if col in df.columns]].to_string(index=False, float_format="{:.3f}".format))
    return df


# %%



def available_anonymization_methods() -> list[str]:
    """Return anonymization methods with complete outputs for the pilot."""
    forced_methods = os.environ.get("EVAL_ANON_METHODS", "").strip()
    if forced_methods:
        methods = [method.strip() for method in forced_methods.split(",") if method.strip()]
        print(f"Using explicit anonymization methods from EVAL_ANON_METHODS: {methods}")
        return methods

    methods = list(METHODS.keys())
    if (ANON_DIR / "deepprivacy2").exists():
        methods.append("deepprivacy2")

    available = []
    for method in methods:
        method_dir = ANON_DIR / method
        if not method_dir.exists():
            continue
        complete = sum(1 for stem in all_stems if anonymization_complete(method, stem))
        if complete == len(all_stems):
            available.append(method)
        else:
            print(f"Skipping incomplete anonymization method {method}: {complete}/{len(all_stems)} complete")
    return available
