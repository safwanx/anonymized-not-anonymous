#!/usr/bin/env python3
# Evaluate person re-ID privacy leakage with OSNet.

import importlib.util

from pipeline_common import *

ANON_METHODS_EVAL = available_anonymization_methods()
print(f"Anonymization methods available for evaluation: {ANON_METHODS_EVAL}")

# 5.2 - Person re-ID evaluation with OSNet
if importlib.util.find_spec("gdown") is None:
    raise ModuleNotFoundError(
        "torchreid imports gdown during startup. Install it in the active venv with: "
        "python -m pip install gdown"
    )

sys.path.insert(0, str(REPOS / "adversaries" / "deep-person-reid"))
from torchreid.utils import FeatureExtractor

osnet_weights = (
    MODELS
    / "person_reid_osnet"
    / "kaiyangzhou_osnet"
    / "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
)
if not osnet_weights.exists():
    raise FileNotFoundError(f"Missing OSNet weights: {osnet_weights}")

reid_extractor = FeatureExtractor(
    model_name="osnet_x1_0",
    model_path=str(osnet_weights),
    device=DEVICE,
    verbose=True,
)


def extract_reid_embeddings(
    file_list: list[str],
    frame_root: Path,
    desc: str = "",
) -> tuple[np.ndarray, list[int], dict[str, float]]:
    embeddings = []
    person_ids = []
    sampled_frames = 0
    frames_with_person = 0

    for filename in tqdm(file_list, desc=desc, leave=False):
        row = meta_by_filename.get(filename)
        if row is None:
            continue
        stem = row["stem"]
        frames_dir = frame_root / stem
        det_path = DETECT_DIR / f"{stem}.json"
        if not frames_dir.exists() or not det_path.exists():
            continue

        detections = read_json(det_path, {})
        video_embeddings = []
        frame_paths = sorted(frames_dir.glob("*.jpg"))[::REID_FRAME_STEP]

        for frame_path in frame_paths:
            sampled_frames += 1
            det = detections.get(frame_path.stem, {})
            persons_det = det.get("persons", [])
            if not persons_det:
                continue

            image = cv2.imread(str(frame_path))
            if image is None:
                continue

            best_box = max(persons_det, key=lambda box: box[4])
            clipped = clip_box(best_box, image.shape[1], image.shape[0])
            if clipped is None:
                continue
            x1, y1, x2, y2 = clipped
            crop = image[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            with torch.no_grad():
                feature = reid_extractor([crop_rgb])
            video_embeddings.append(feature.detach().cpu().numpy().flatten())
            frames_with_person += 1

        if video_embeddings:
            embeddings.append(np.mean(video_embeddings, axis=0))
            person_ids.append(int(row["person"]))

    stats = {
        "requested": float(len(file_list)),
        "valid": float(len(embeddings)),
        "video_coverage": len(embeddings) / max(len(file_list), 1),
        "sampled_frames": float(sampled_frames),
        "person_frame_coverage": frames_with_person / max(sampled_frames, 1),
    }
    if embeddings:
        return np.stack(embeddings), person_ids, stats
    return np.zeros((0, 512), dtype=np.float32), [], stats


reid_rows = []
for protocol, (gallery_files, probe_files) in EVAL_PROTOCOLS.items():
    gallery_feats, gallery_ids, gallery_stats = extract_reid_embeddings(
        gallery_files,
        FRAMES_DIR,
        desc=f"reid gallery {protocol}",
    )
    for method in ["original"] + ANON_METHODS_EVAL:
        probe_feats, probe_ids, probe_stats = extract_reid_embeddings(
            probe_files,
            frame_root_for_method(method),
            desc=f"reid probe {protocol} {method}",
        )
        metrics = compute_reid_metrics(gallery_feats, gallery_ids, probe_feats, probe_ids)
        metrics.update(
            {
                "attack": "reid",
                "protocol": protocol,
                "method": method,
                "gallery_requested": len(gallery_files),
                "probe_requested": len(probe_files),
                "gallery_coverage": gallery_stats["video_coverage"],
                "probe_coverage": probe_stats["video_coverage"],
                "probe_person_frame_coverage": probe_stats["person_frame_coverage"],
                "coverage_adjusted_rank1": metrics["rank1"] * probe_stats["video_coverage"],
            }
        )
        reid_rows.append(metrics)

df_reid = save_attack_results("person_reid", reid_rows)
del reid_extractor
free_gpu()

# %%
