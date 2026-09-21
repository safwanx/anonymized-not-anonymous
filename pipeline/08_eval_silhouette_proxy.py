#!/usr/bin/env python3
# Evaluate silhouette/gait proxy privacy leakage.

from pipeline_common import *

ANON_METHODS_EVAL = available_anonymization_methods()
print(f"Anonymization methods available for evaluation: {ANON_METHODS_EVAL}")

# 5.4 - Silhouette/gait proxy evaluation
SIL_H, SIL_W = 64, 44


def extract_silhouette_features(
    file_list: list[str],
    frame_root: Path,
    desc: str = "",
    max_videos: int = SIL_MAX_EVAL,
) -> tuple[np.ndarray, list[int], dict[str, float]]:
    selected_files = safe_limit(file_list, max_videos) if max_videos else file_list
    embeddings = []
    person_ids = []
    sampled_frames = 0
    frames_with_person = 0

    for filename in tqdm(selected_files, desc=desc, leave=False):
        row = meta_by_filename.get(filename)
        if row is None:
            continue
        stem = row["stem"]
        frames_dir = frame_root / stem
        det_path = DETECT_DIR / f"{stem}.json"
        if not frames_dir.exists() or not det_path.exists():
            continue

        detections = read_json(det_path, {})
        silhouettes = []
        for frame_path in sorted(frames_dir.glob("*.jpg"))[::SIL_FRAME_STEP]:
            sampled_frames += 1
            det = detections.get(frame_path.stem, {})
            persons_det = det.get("persons", [])
            if not persons_det:
                continue
            image = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            best_box = max(persons_det, key=lambda box: box[4])
            clipped = clip_box(best_box, image.shape[1], image.shape[0])
            if clipped is None:
                continue
            x1, y1, x2, y2 = clipped
            crop = image[y1:y2, x1:x2]
            _, binary = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            silhouette = cv2.resize(binary, (SIL_W, SIL_H))
            silhouettes.append(silhouette.flatten().astype(np.float32) / 255.0)
            frames_with_person += 1

        if len(silhouettes) >= 5:
            sil_arr = np.asarray(silhouettes)
            feature = np.concatenate([sil_arr.mean(axis=0), sil_arr.std(axis=0)])
            embeddings.append(feature)
            person_ids.append(int(row["person"]))

    stats = {
        "requested": float(len(file_list)),
        "used": float(len(selected_files)),
        "valid": float(len(embeddings)),
        "video_coverage": len(embeddings) / max(len(selected_files), 1),
        "person_frame_coverage": frames_with_person / max(sampled_frames, 1),
    }
    if embeddings:
        return np.stack(embeddings), person_ids, stats
    return np.zeros((0, SIL_H * SIL_W * 2), dtype=np.float32), [], stats


sil_rows = []
for protocol, (gallery_files, probe_files) in EVAL_PROTOCOLS.items():
    gallery_feats, gallery_ids, gallery_stats = extract_silhouette_features(
        gallery_files,
        FRAMES_DIR,
        desc=f"sil gallery {protocol}",
    )
    for method in ["original"] + ANON_METHODS_EVAL:
        probe_feats, probe_ids, probe_stats = extract_silhouette_features(
            probe_files,
            frame_root_for_method(method),
            desc=f"sil probe {protocol} {method}",
        )
        metrics = compute_reid_metrics(gallery_feats, gallery_ids, probe_feats, probe_ids)
        metrics.update(
            {
                "attack": "silhouette_proxy",
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
        sil_rows.append(metrics)

df_sil = save_attack_results("silhouette_proxy", sil_rows)

# %% [markdown]
