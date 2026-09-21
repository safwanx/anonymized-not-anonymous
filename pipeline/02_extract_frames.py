#!/usr/bin/env python3
# Extract pilot RGB frames with completion manifests.

from pipeline_common import *

# 2.2 - Extract frames
JPG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, FRAME_JPEG_QUALITY]

frame_todo = build_frame_todo()
skipped_already_complete = 0

for stem in tqdm(frame_todo, desc="Extracting frames"):
    if frame_extraction_complete(stem):
        skipped_already_complete += 1
        continue

    video_path = rgb_paths[stem]
    out_dir = FRAMES_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    for old_frame in out_dir.glob("*.jpg"):
        old_frame.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        write_json(
            frame_manifest_path(stem),
            {"status": "failed", "reason": "VideoCapture open failed", "video": str(video_path)},
        )
        continue

    raw_frames = 0
    extracted = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if raw_frames % FRAME_STRIDE == 0:
            out_path = out_dir / f"{raw_frames:06d}.jpg"
            if cv2.imwrite(str(out_path), frame, JPG_PARAMS):
                extracted += 1
        raw_frames += 1
    cap.release()

    status = "complete" if extracted > 0 else "failed"
    write_json(
        frame_manifest_path(stem),
        {
            "status": status,
            "video": str(video_path),
            "raw_frames": raw_frames,
            "extracted_frames": extracted,
            "stride": FRAME_STRIDE,
            "jpeg_quality": FRAME_JPEG_QUALITY,
        },
    )

if skipped_already_complete:
    print(f"Skipped already extracted videos during loop: {skipped_already_complete}")

print("Frame extraction section complete.")

# %%
# 2.3 - Verify frame extraction
frame_counts = {stem: count_jpgs(FRAMES_DIR / stem) for stem in all_stems}
completed_frames = [count for stem, count in frame_counts.items() if frame_extraction_complete(stem)]
failed_frames = [stem for stem in all_stems if not frame_extraction_complete(stem)]

print(f"Completed videos: {len(completed_frames)}/{len(all_stems)}")
print(f"Failed/incomplete: {len(failed_frames)}")
if completed_frames:
    print(
        f"Frames/video: min={min(completed_frames)}, max={max(completed_frames)}, "
        f"mean={sum(completed_frames) / len(completed_frames):.1f}"
    )
if failed_frames:
    print("First incomplete stems:", failed_frames[:5])

# %%
# 2.4 - Visualize a frame strip
sample_stem = next((stem for stem in all_stems if frame_extraction_complete(stem)), all_stems[0])
sample_frames = sorted((FRAMES_DIR / sample_stem).glob("*.jpg"))
if sample_frames:
    selected_frames = sample_frame_paths(FRAMES_DIR / sample_stem, 8)
    fig, axes = plt.subplots(1, len(selected_frames), figsize=(16, 2.5))
    for ax, frame_path in zip(np.atleast_1d(axes), selected_frames):
        img = cv2.imread(str(frame_path))
        if img is not None:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis("off")
    plt.suptitle(f"Frame strip: {sample_stem}", fontsize=10)
    plt.tight_layout()
    plt.show()

# %% [markdown]
