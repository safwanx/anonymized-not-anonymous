#!/usr/bin/env python3
# Generate simple anonymization baselines and optional DeepPrivacy2 outputs.

from pipeline_common import *

# 4.2 - Run simple anonymization baselines
JPG_PARAMS_ANON = [cv2.IMWRITE_JPEG_QUALITY, ANON_JPEG_QUALITY]

for method_name, method_fn in METHODS.items():
    todo = [
        stem
        for stem in all_stems
        if detection_complete(stem) and not anonymization_complete(method_name, stem)
    ]
    print(f"{method_name}: {len(all_stems) - len(todo)} complete/irrelevant, {len(todo)} to process")

    for stem in tqdm(todo, desc=f"Anonymizing {method_name}"):
        detections = read_json(DETECT_DIR / f"{stem}.json", {})
        out_dir = ANON_DIR / method_name / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        for frame_path in sorted((FRAMES_DIR / stem).glob("*.jpg")):
            out_path = out_dir / frame_path.name
            if out_path.exists():
                continue
            image = cv2.imread(str(frame_path))
            if image is None:
                continue
            det = detections.get(frame_path.stem, {"persons": [], "faces": []})
            anon_image = method_fn(image, det)
            cv2.imwrite(str(out_path), anon_image, JPG_PARAMS_ANON)

print("Simple anonymization section complete.")

# %%
# 4.3 - DeepPrivacy2 optional setup
DP2_AVAILABLE = False
dp2_repo = REPOS / "privacy_methods" / "deep_privacy2"
dp2_config = dp2_repo / "configs" / "anonymizers" / "FB_cse.py"
dp2_error_log = LOGS_DIR / "deepprivacy2_errors.jsonl"

try:
    sys.path.insert(0, str(dp2_repo))
    from dp2 import utils as dp2_utils
    from tops.config import instantiate as tops_instantiate

    print("Loading DeepPrivacy2 anonymizer...")
    cfg = dp2_utils.load_config(str(dp2_config))
    cfg.detector.score_threshold = 0.3
    dp2_anonymizer = tops_instantiate(cfg.anonymizer, load_cache=False)
    dp2_synthesis_kwargs = {
        "amp": DEVICE == "cuda",
        "multi_modal_truncation": False,
        "truncation_value": 0,
    }
    DP2_AVAILABLE = True
    print("DeepPrivacy2 loaded.")
except Exception as exc:
    print(f"DeepPrivacy2 not available: {exc}")
    print("Skipping DeepPrivacy2. This is acceptable for the first pilot pass.")

# %%
# 4.4 - Run DeepPrivacy2 if available
if DP2_AVAILABLE:
    dp2_method = "deepprivacy2"
    dp2_dir = ANON_DIR / dp2_method
    todo = [
        stem
        for stem in all_stems
        if frame_extraction_complete(stem) and not anonymization_complete(dp2_method, stem)
    ]
    print(f"DeepPrivacy2: {len(all_stems) - len(todo)} complete/irrelevant, {len(todo)} to process")

    for stem in tqdm(todo, desc="DeepPrivacy2"):
        out_dir = dp2_dir / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        for frame_path in sorted((FRAMES_DIR / stem).glob("*.jpg")):
            out_path = out_dir / frame_path.name
            if out_path.exists():
                continue
            image = cv2.imread(str(frame_path))
            if image is None:
                continue
            try:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_t = dp2_utils.im2torch(image_rgb, to_float=False, normalize=False)[0]
                anon_t = dp2_anonymizer(image_t, **dp2_synthesis_kwargs)
                anon_np = dp2_utils.im2numpy(anon_t)
                anon_bgr = cv2.cvtColor(anon_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_path), anon_bgr, JPG_PARAMS_ANON)
            except Exception as exc:
                append_jsonl(
                    dp2_error_log,
                    {"stem": stem, "frame": frame_path.name, "error": str(exc)},
                )
                # Do not write the original frame as a fake anonymized result.

    del dp2_anonymizer
    free_gpu()
    print("DeepPrivacy2 section complete.")
else:
    print("DeepPrivacy2 skipped.")

# %%
# 4.5 - Determine available anonymization methods
ANON_METHODS_EVAL = [
    method
    for method in list(METHODS.keys()) + (["deepprivacy2"] if DP2_AVAILABLE else [])
    if (ANON_DIR / method).exists()
]
print(f"Anonymization methods available for evaluation: {ANON_METHODS_EVAL}")

# %%
# 4.6 - Visualize anonymization outputs
vis_stem = next((stem for stem in all_stems if frame_extraction_complete(stem)), None)
if vis_stem:
    vis_frames = sorted((FRAMES_DIR / vis_stem).glob("*.jpg"))
    if len(vis_frames) >= 1:
        vis_frame = vis_frames[min(5, len(vis_frames) - 1)]
        available_methods = [
            method
            for method in ANON_METHODS_EVAL
            if (ANON_DIR / method / vis_stem / vis_frame.name).exists()
        ]
        ncols = 1 + len(available_methods)
        fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
        axes = np.atleast_1d(axes)
        original = cv2.imread(str(vis_frame))
        if original is not None:
            axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0].set_title("original", fontsize=9)
        axes[0].axis("off")

        for ax, method in zip(axes[1:], available_methods):
            anon_path = ANON_DIR / method / vis_stem / vis_frame.name
            image = cv2.imread(str(anon_path))
            if image is not None:
                ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax.set_title(method, fontsize=9)
            ax.axis("off")

        plt.suptitle(f"Anonymization comparison: {vis_stem}", fontsize=10)
        plt.tight_layout()
        plt.savefig(str(OUTPUT_ROOT / "vis_anonymization.png"), dpi=120)
        plt.show()

# %% [markdown]
