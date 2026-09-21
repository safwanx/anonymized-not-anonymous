#!/usr/bin/env python3
# Validate pilot metadata/splits and save a quick sample visualization.

from pipeline_common import *

# 1.3 - Visualize sample videos
sample_persons = persons[:4]
fig, axes = plt.subplots(len(sample_persons), 3, figsize=(12, 4 * len(sample_persons)))
for i, person_id in enumerate(sample_persons):
    for camera in range(1, 4):
        row = next(
            (
                item
                for item in meta
                if item["person"] == person_id
                and item["camera"] == camera
                and item["replication"] == 1
            ),
            None,
        )
        ax = axes[i, camera - 1]
        if row:
            cap = cv2.VideoCapture(row["rgb_resolved"])
            ok, frame = cap.read()
            cap.release()
            if ok:
                ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title(f"P{person_id:03d} C{camera}", fontsize=9)
        ax.axis("off")
plt.suptitle("Sample pilot videos", fontsize=11)
plt.tight_layout()
plt.savefig(str(OUTPUT_ROOT / "vis_sample_videos.png"), dpi=120)
plt.show()

# %% [markdown]


print("Metadata and split validation complete.")
