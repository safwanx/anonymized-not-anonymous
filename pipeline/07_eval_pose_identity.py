#!/usr/bin/env python3
# Evaluate identity leakage from native NTU skeletons.

from pipeline_common import *

# 5.3 - Pose-based identity leakage using native NTU skeletons
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler


def parse_ntu_skeleton(skeleton_path: Path) -> np.ndarray | None:
    with skeleton_path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()

    idx = 0
    num_frames = int(lines[idx])
    idx += 1
    frames = []

    for _ in range(num_frames):
        num_bodies = int(lines[idx])
        idx += 1
        bodies = []
        for _ in range(num_bodies):
            idx += 1
            num_joints = int(lines[idx])
            idx += 1
            joints = []
            for _ in range(num_joints):
                parts = lines[idx].split()
                idx += 1
                joints.append([float(parts[0]), float(parts[1]), float(parts[2])])
            bodies.append(joints)
        if bodies:
            frames.append(bodies[0])

    if not frames:
        return None

    seq = np.asarray(frames, dtype=np.float32)
    seq = seq - seq[:, :1, :]  # root-relative to reduce camera/setup position leakage
    scale = np.std(seq) + 1e-6
    return seq / scale


def skeleton_feature_for_file(filename: str, max_frames: int = 50) -> np.ndarray | None:
    row = meta_by_filename.get(filename)
    if row is None or not row["skeleton_resolved"]:
        return None
    seq = parse_ntu_skeleton(Path(row["skeleton_resolved"]))
    if seq is None or len(seq) < 5:
        return None
    if len(seq) > max_frames:
        seq = seq[:max_frames]
    elif len(seq) < max_frames:
        pad = np.zeros((max_frames - len(seq), 25, 3), dtype=np.float32)
        seq = np.concatenate([seq, pad], axis=0)
    return seq.reshape(-1)


def build_skeleton_xy(files: list[str]) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for filename in tqdm(files, desc="skeleton features", leave=False):
        feature = skeleton_feature_for_file(filename)
        if feature is None:
            continue
        features.append(feature)
        labels.append(int(meta_by_filename[filename]["person"]))
    if features:
        return np.stack(features), np.asarray(labels)
    return np.zeros((0, 50 * 25 * 3), dtype=np.float32), np.asarray([])


pose_rows = []
for protocol, (gallery_files, probe_files) in EVAL_PROTOCOLS.items():
    x_train, y_train = build_skeleton_xy(gallery_files)
    x_test, y_test = build_skeleton_xy(probe_files)
    if len(x_train) == 0 or len(x_test) == 0:
        continue

    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    known = np.isin(y_test, encoder.classes_)
    x_test = x_test[known]
    y_test = y_test[known]
    y_test_enc = encoder.transform(y_test)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    clf = MLPClassifier(
        hidden_layer_sizes=(512, 256),
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=RANDOM_SEED,
        verbose=False,
    )
    clf.fit(x_train_scaled, y_train_enc)
    pred = clf.predict(x_test_scaled)

    pose_rows.append(
        {
            "attack": "pose_native_skeleton",
            "protocol": protocol,
            "method": "native_skeleton",
            "train_samples": len(x_train),
            "test_samples": len(x_test),
            "num_persons": len(encoder.classes_),
            "accuracy": accuracy_score(y_test_enc, pred),
            "balanced_accuracy": balanced_accuracy_score(y_test_enc, pred),
            "chance": 1 / max(len(encoder.classes_), 1),
        }
    )

df_pose = pd.DataFrame(pose_rows)
df_pose.to_csv(FEATURES_DIR / "pose_identity_results.csv", index=False)
print(df_pose.to_string(index=False, float_format="{:.3f}".format))

# %%
