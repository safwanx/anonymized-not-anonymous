#!/usr/bin/env python3
# Detect people and faces in extracted pilot frames.

from pipeline_common import *

# 3.1 - Load detection models
from insightface.app import FaceAnalysis
from ultralytics import YOLO

print(f"Loading YOLO model: {YOLO_MODEL}")
yolo = YOLO(YOLO_MODEL)
yolo.to(DEVICE)

print("Loading InsightFace buffalo_l for face detection")
insightface_root = ensure_insightface_root("buffalo_l")
face_app = FaceAnalysis(
    name="buffalo_l",
    root=insightface_root,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
face_app.prepare(ctx_id=0 if DEVICE == "cuda" else -1, det_size=(640, 640))
print("Detection models loaded.")

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


det_todo = build_detection_todo()

# %%
# 3.3 - Run detection
for stem in tqdm(det_todo, desc="Detecting"):
    frames_dir = FRAMES_DIR / stem
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        continue

    results: dict[str, Any] = {}
    for batch_start in range(0, len(frame_paths), DETECTION_BATCH_SIZE):
        batch_paths = frame_paths[batch_start : batch_start + DETECTION_BATCH_SIZE]
        loaded = [(path, cv2.imread(str(path))) for path in batch_paths]
        loaded = [(path, image) for path, image in loaded if image is not None]
        if not loaded:
            continue

        paths, images = zip(*loaded)
        yolo_out = yolo.predict(
            list(images),
            classes=[0],
            conf=0.3,
            verbose=False,
            device=DEVICE,
        )

        for frame_path, image, ydet in zip(paths, images, yolo_out):
            persons = []
            if ydet.boxes is not None:
                for box in ydet.boxes:
                    xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
                    conf = float(box.conf[0].detach().cpu())
                    persons.append([round(value, 1) for value in xyxy] + [round(conf, 3)])

            faces = []
            try:
                for face in face_app.get(image):
                    bbox = face.bbox.tolist()
                    faces.append([round(value, 1) for value in bbox] + [round(float(face.det_score), 3)])
            except Exception as exc:
                append_jsonl(LOGS_DIR / "face_detection_errors.jsonl", {"stem": stem, "frame": frame_path.name, "error": str(exc)})

            results[frame_path.stem] = {"persons": persons, "faces": faces}

    write_json(DETECT_DIR / f"{stem}.json", results)

print("Detection section complete.")

# %%
# 3.4 - Detection statistics
stats_rows = []
for stem in all_stems:
    det = read_json(DETECT_DIR / f"{stem}.json", {})
    if not det:
        continue
    info = parse_filename(stem) or {}
    camera = int(info.get("camera", 0))
    for frame_key, item in det.items():
        stats_rows.append(
            {
                "stem": stem,
                "camera": camera,
                "frame": frame_key,
                "num_persons": len(item.get("persons", [])),
                "num_faces": len(item.get("faces", [])),
                "has_person": bool(item.get("persons", [])),
                "has_face": bool(item.get("faces", [])),
            }
        )

df_det_stats = pd.DataFrame(stats_rows)
if not df_det_stats.empty:
    coverage = df_det_stats.groupby("camera").agg(
        frames=("frame", "count"),
        person_frames=("has_person", "sum"),
        face_frames=("has_face", "sum"),
        person_dets=("num_persons", "sum"),
        face_dets=("num_faces", "sum"),
    )
    coverage["person_frame_coverage"] = coverage["person_frames"] / coverage["frames"]
    coverage["face_frame_coverage"] = coverage["face_frames"] / coverage["frames"]
    print(coverage)
    coverage.to_csv(FEATURES_DIR / "detection_coverage_by_camera.csv")

# %%
# 3.5 - Visualize detections
sample_stem = next((stem for stem in all_stems if detection_complete(stem)), None)
if sample_stem:
    det_data = read_json(DETECT_DIR / f"{sample_stem}.json", {})
    frame_keys = sorted(det_data.keys())[:4]
    if frame_keys:
        fig, axes = plt.subplots(1, len(frame_keys), figsize=(16, 4))
        for ax, frame_key in zip(np.atleast_1d(axes), frame_keys):
            image = cv2.imread(str(FRAMES_DIR / sample_stem / f"{frame_key}.jpg"))
            if image is None:
                ax.axis("off")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            for box in det_data[frame_key]["persons"]:
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for box in det_data[frame_key]["faces"]:
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            ax.imshow(image)
            ax.set_title(f"Frame {frame_key}", fontsize=9)
            ax.axis("off")
        plt.suptitle(f"Detections: {sample_stem}", fontsize=10)
        plt.tight_layout()
        plt.savefig(str(OUTPUT_ROOT / "vis_detections.png"), dpi=120)
        plt.show()

# %%
# 3.6 - Free detection models
del yolo, face_app
free_gpu()
print("Detection models freed.")

# %% [markdown]
