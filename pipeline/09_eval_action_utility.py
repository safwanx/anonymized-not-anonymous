#!/usr/bin/env python3
# Evaluate lightweight RGB action-utility proxy.

from pipeline_common import *

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

ANON_METHODS_EVAL = available_anonymization_methods()
print(f"Anonymization methods available for evaluation: {ANON_METHODS_EVAL}")

# 6.1 - RGB utility proxy feature extraction
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline


def video_proxy_feature(filename: str, frame_root: Path) -> np.ndarray | None:
    row = meta_by_filename.get(filename)
    if row is None:
        return None
    frame_dir = frame_root / row["stem"]
    frame_paths = sample_frame_paths(frame_dir, UTILITY_MAX_FRAMES)
    if not frame_paths:
        return None

    frames = []
    for frame_path in frame_paths:
        image = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        image = cv2.resize(image, (UTILITY_FRAME_SIZE, UTILITY_FRAME_SIZE))
        frames.append(image.astype(np.float32) / 255.0)
    if not frames:
        return None

    arr = np.stack(frames)
    mean_frame = arr.mean(axis=0).flatten()
    std_frame = arr.std(axis=0).flatten()
    if len(arr) > 1:
        diff_frame = np.abs(np.diff(arr, axis=0)).mean(axis=0).flatten()
    else:
        diff_frame = np.zeros_like(mean_frame)
    return np.concatenate([mean_frame, std_frame, diff_frame])


def build_action_xy(files: list[str], frame_root: Path) -> tuple[np.ndarray, np.ndarray, int]:
    features = []
    labels = []
    missing = 0
    for filename in tqdm(files, desc="action proxy features", leave=False):
        feature = video_proxy_feature(filename, frame_root)
        if feature is None:
            missing += 1
            continue
        features.append(feature)
        labels.append(int(meta_by_filename[filename]["action"]))
    if features:
        return np.stack(features), np.asarray(labels), missing
    dim = UTILITY_FRAME_SIZE * UTILITY_FRAME_SIZE * 3
    return np.zeros((0, dim), dtype=np.float32), np.asarray([]), missing


def topk_accuracy_from_scores(scores: np.ndarray, classes: np.ndarray, y_true: np.ndarray, k: int) -> float:
    if len(y_true) == 0:
        return 0.0
    top_indices = np.argsort(scores, axis=1)[:, -k:]
    top_labels = classes[top_indices]
    return float(np.mean([truth in row for truth, row in zip(y_true, top_labels)]))


utility_rows = []
utility_train_files = gallery_xv_files
x_train, y_train, train_missing = build_action_xy(utility_train_files, FRAMES_DIR)

if len(x_train) > 0:
    action_clf = make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="log_loss",
            max_iter=1000,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    )
    action_clf.fit(x_train, y_train)

    for protocol, probe_files in [
        ("crossview_cam2", probe_xv_c2_files),
        ("crossview_cam3", probe_xv_c3_files),
    ]:
        for method in ["original"] + ANON_METHODS_EVAL:
            x_test, y_test, missing = build_action_xy(probe_files, frame_root_for_method(method))
            if len(x_test) == 0:
                continue
            pred = action_clf.predict(x_test)
            scores = action_clf.decision_function(x_test)
            top1 = accuracy_score(y_test, pred)
            top5 = topk_accuracy_from_scores(scores, action_clf.classes_, y_test, k=min(5, len(action_clf.classes_)))
            mean_class = balanced_accuracy_score(y_test, pred)
            utility_rows.append(
                {
                    "protocol": protocol,
                    "method": method,
                    "train_samples": len(x_train),
                    "test_samples": len(x_test),
                    "missing_test_features": missing,
                    "top1": top1,
                    "top5": top5,
                    "mean_class_accuracy": mean_class,
                    "note": "lightweight RGB proxy, not final action model",
                }
            )

df_utility = pd.DataFrame(utility_rows)
df_utility.to_csv(FEATURES_DIR / "action_utility_proxy_results.csv", index=False)
if not df_utility.empty:
    print(df_utility.to_string(index=False, float_format="{:.3f}".format))
else:
    print("Action utility proxy did not run because no frame features were available.")

# %% [markdown]
