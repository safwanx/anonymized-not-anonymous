#!/usr/bin/env python3
"""Identity-pool scaling analysis (stage 21).

Addresses the 'protocol coverage' limitation. The protocol fixes a 38-subject
identity pool, and re-ID/face accuracy at 38 subjects is an upper bound relative
to the full NTU RGB+D 120 pool (106 subjects). Rather than assert this, we
measure it: how does identity leakage scale with the gallery identity-pool size?

For each adversary (reid, face) and split, we subsample the gallery to N unique
identities, restrict probes to those same identities (closed-set re-ID), and
recompute rank-1/rank-5/mAP over many random seeds. We then fit rank1 against
log(N) and extrapolate to the full pool, turning the caveat into a measured
scaling law with quantified uncertainty.

Embeddings are extracted/loaded ONCE per (adversary, method, split); the
subsample loop is pure CPU on cached arrays, so the sweep itself is cheap. The
only GPU cost is the one-time re-ID embedding pass (face embeddings are reused
from the stage-05 cache).

Run after stages 05/06 have populated frames + the face cache:

    cd $ACCV/pipeline
    python 21_eval_identity_pool_scaling.py \
        --adversaries reid,face --methods original \
        --sizes 5,10,15,20,25,30,38 --seeds 30 --full-pool 106
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from pipeline_common import *

FACE_CACHE_DIR = FEATURES_DIR / "face_recognition_cache"
REID_CACHE_DIR = FEATURES_DIR / "reid_scaling_cache"

# Splits worth sweeping: need enough distinct identities to subsample. The
# cross-range split has only 3 bridge subjects, so a pool sweep is undefined
# there; we report it as a single fixed point in the paper instead.
SWEEP_SPLITS = ["crossview_cam2", "crossview_cam3", "crosssetup"]
MIN_POOL_FOR_SWEEP = 8
# Draw the fit + full-pool extrapolation only for splits measured to at least
# this many identities; under-sampled splits are shown as points only.
EXTRAPOLATE_MIN_POOL = 30


# --------------------------------------------------------------------------- #
# Embedding access: face from the stage-05 npz cache, reid extracted + cached.
# --------------------------------------------------------------------------- #
_reid_extractor = None


def _load_face_embedding(method: str, filename: str) -> np.ndarray | None:
    stem = stem_of(filename)
    path = FACE_CACHE_DIR / method / f"{stem}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    emb = data["embedding"].astype(np.float32, copy=False)
    return emb if emb.shape == (512,) else None


def _get_reid_extractor():
    global _reid_extractor
    if _reid_extractor is not None:
        return _reid_extractor
    import sys as _sys

    _sys.path.insert(0, str(REPOS / "adversaries" / "deep-person-reid"))
    from torchreid.utils import FeatureExtractor

    weights = (
        MODELS
        / "person_reid_osnet"
        / "kaiyangzhou_osnet"
        / "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
    )
    if not weights.exists():
        raise FileNotFoundError(f"Missing OSNet weights: {weights}")
    _reid_extractor = FeatureExtractor(
        model_name="osnet_x1_0", model_path=str(weights), device=DEVICE, verbose=False
    )
    return _reid_extractor


def _compute_reid_embedding(method: str, filename: str) -> np.ndarray | None:
    row = meta_by_filename.get(filename)
    if row is None:
        return None
    stem = row["stem"]
    frames_dir = frame_root_for_method(method) / stem
    det_path = DETECT_DIR / f"{stem}.json"
    if not frames_dir.exists() or not det_path.exists():
        return None

    detections = read_json(det_path, {})
    extractor = _get_reid_extractor()
    video_embeddings = []
    for frame_path in sorted(frames_dir.glob("*.jpg"))[::REID_FRAME_STEP]:
        det = detections.get(frame_path.stem, {})
        persons_det = det.get("persons", [])
        if not persons_det:
            continue
        image = cv2.imread(str(frame_path))
        if image is None:
            continue
        clipped = clip_box(max(persons_det, key=lambda b: b[4]), image.shape[1], image.shape[0])
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        crop_rgb = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            feature = extractor([crop_rgb])
        video_embeddings.append(feature.detach().cpu().numpy().flatten())

    if not video_embeddings:
        return None
    return np.mean(video_embeddings, axis=0).astype(np.float32)


def _load_reid_embedding(method: str, filename: str) -> np.ndarray | None:
    stem = stem_of(filename)
    path = REID_CACHE_DIR / method / f"{stem}.npz"
    if path.exists():
        emb = np.load(path)["embedding"].astype(np.float32, copy=False)
        return emb if emb.shape == (512,) else None
    emb = _compute_reid_embedding(method, filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, embedding=emb if emb is not None else np.zeros((0,), dtype=np.float32)
    )
    return emb


def build_matrix(adversary: str, method: str, file_list: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return (feats[N,512], person_ids[N]) for the videos with a valid embedding."""
    loader = _load_reid_embedding if adversary == "reid" else _load_face_embedding
    feats, ids = [], []
    for filename in tqdm(file_list, desc=f"{adversary}:{method}", leave=False):
        row = meta_by_filename.get(filename)
        if row is None:
            continue
        emb = loader(method, filename)
        if emb is None:
            continue
        feats.append(emb)
        ids.append(int(row["person"]))
    if not feats:
        return np.zeros((0, 512), dtype=np.float32), np.zeros((0,), dtype=int)
    return np.stack(feats), np.asarray(ids, dtype=int)


# --------------------------------------------------------------------------- #
# Subsample-and-score
# --------------------------------------------------------------------------- #
def subsample_score(
    g_feats: np.ndarray,
    g_ids: np.ndarray,
    p_feats: np.ndarray,
    p_ids: np.ndarray,
    pool: int,
    seed: int,
) -> dict[str, float] | None:
    """Restrict gallery+probe to `pool` identities and score closed-set re-ID."""
    shared = np.array(sorted(set(g_ids.tolist()) & set(p_ids.tolist())))
    if len(shared) < pool:
        return None
    rng = np.random.default_rng(seed)
    chosen = set(rng.choice(shared, size=pool, replace=False).tolist())

    gm = np.array([pid in chosen for pid in g_ids])
    pm = np.array([pid in chosen for pid in p_ids])
    if gm.sum() == 0 or pm.sum() == 0:
        return None
    return compute_reid_metrics(g_feats[gm], g_ids[gm].tolist(), p_feats[pm], p_ids[pm].tolist())


def fit_log_curve(sizes: np.ndarray, rank1: np.ndarray, full_pool: int) -> dict[str, float]:
    """Fit rank1 = a + b*ln(N); return slope, intercept, R^2, and value at full_pool."""
    if len(sizes) < 2:
        return {}
    x = np.log(sizes)
    b, a = np.polyfit(x, rank1, 1)  # slope, intercept
    pred = a + b * x
    ss_res = float(np.sum((rank1 - pred) ** 2))
    ss_tot = float(np.sum((rank1 - rank1.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {
        "fit_slope": float(b),
        "fit_intercept": float(a),
        "fit_r2": float(r2),
        "extrapolated_rank1_full_pool": float(max(0.0, a + b * np.log(full_pool))),
        "full_pool": float(full_pool),
    }


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adversaries", default="reid,face")
    p.add_argument("--methods", default=os.environ.get("EVAL_ANON_METHODS", "original"))
    p.add_argument("--sizes", default="5,10,15,20,25,30,38")
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--full-pool", type=int, default=106, help="Full NTU RGB+D 120 subject count.")
    p.add_argument("--splits", default=",".join(SWEEP_SPLITS))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    adversaries = [a.strip() for a in args.adversaries.split(",") if a.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    sizes = sorted(int(s) for s in args.sizes.split(",") if s.strip())
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    print(f"Adversaries : {adversaries}")
    print(f"Methods     : {methods}")
    print(f"Pool sizes  : {sizes}")
    print(f"Seeds       : {args.seeds}")
    print(f"Full pool   : {args.full_pool}")
    print(f"Splits      : {splits}")

    raw_rows: list[dict] = []
    agg_rows: list[dict] = []
    fit_rows: list[dict] = []

    for adversary in adversaries:
        for split in splits:
            if split not in EVAL_PROTOCOLS:
                print(f"  skip unknown split {split}")
                continue
            gallery_files, probe_files = EVAL_PROTOCOLS[split]
            g_feats, g_ids = build_matrix(adversary, "original", gallery_files)
            n_pool = len(set(g_ids.tolist()))
            if n_pool < MIN_POOL_FOR_SWEEP:
                print(f"  skip {adversary}/{split}: only {n_pool} gallery identities")
                continue
            print(f"\n[{adversary}/{split}] gallery identities available: {n_pool}")

            for method in methods:
                p_feats, p_ids = build_matrix(adversary, method, probe_files)
                shared = sorted(set(g_ids.tolist()) & set(p_ids.tolist()))
                usable_sizes = [n for n in sizes if n <= len(shared)]
                if not usable_sizes:
                    print(f"  {method}: no usable pool sizes (shared identities={len(shared)})")
                    continue

                mean_by_n: list[tuple[int, float]] = []
                for n in usable_sizes:
                    vals = {"rank1": [], "rank5": [], "mAP": []}
                    n_seeds = 1 if n >= len(shared) else args.seeds
                    for seed in range(n_seeds):
                        m = subsample_score(g_feats, g_ids, p_feats, p_ids, n, seed)
                        if m is None:
                            continue
                        for k in vals:
                            vals[k].append(m[k])
                        raw_rows.append(
                            {
                                "adversary": adversary,
                                "split": split,
                                "method": method,
                                "pool_size": n,
                                "seed": seed,
                                "rank1": m["rank1"],
                                "rank5": m["rank5"],
                                "mAP": m["mAP"],
                            }
                        )
                    if not vals["rank1"]:
                        continue
                    row = {
                        "adversary": adversary,
                        "split": split,
                        "method": method,
                        "pool_size": n,
                        "n_seeds": len(vals["rank1"]),
                    }
                    for k in vals:
                        row[f"{k}_mean"] = float(np.mean(vals[k]))
                        row[f"{k}_std"] = float(np.std(vals[k]))
                    agg_rows.append(row)
                    mean_by_n.append((n, row["rank1_mean"]))
                    print(
                        f"  {method:11s} N={n:3d}  rank1={row['rank1_mean']:.3f}"
                        f"±{row['rank1_std']:.3f}  mAP={row['mAP_mean']:.3f}"
                    )

                if len(mean_by_n) >= 2:
                    ns = np.array([n for n, _ in mean_by_n], dtype=float)
                    r1 = np.array([v for _, v in mean_by_n], dtype=float)
                    fit = fit_log_curve(ns, r1, args.full_pool)
                    if fit:
                        fit.update({"adversary": adversary, "split": split, "method": method,
                                    "max_pool_measured": int(ns.max()),
                                    "rank1_at_max_pool": float(r1[ns.argmax()])})
                        fit_rows.append(fit)
                        print(
                            f"  -> fit rank1={fit['fit_intercept']:.3f}{fit['fit_slope']:+.3f}*ln(N)"
                            f"  R^2={fit['fit_r2']:.3f}  extrapolated@{args.full_pool}="
                            f"{fit['extrapolated_rank1_full_pool']:.3f}"
                        )

    if raw_rows:
        pd.DataFrame(raw_rows).to_csv(FEATURES_DIR / "identity_pool_scaling_raw.csv", index=False)
    if agg_rows:
        pd.DataFrame(agg_rows).to_csv(FEATURES_DIR / "identity_pool_scaling.csv", index=False)
    if fit_rows:
        pd.DataFrame(fit_rows).to_csv(FEATURES_DIR / "identity_pool_scaling_fit.csv", index=False)
        _plot(agg_rows, fit_rows, args.full_pool)
    print(f"\nSaved scaling CSVs to {FEATURES_DIR}")
    return 0


def _plot(agg_rows: list[dict], fit_rows: list[dict], full_pool: int) -> None:
    df = pd.DataFrame(agg_rows)
    fits = {(r["adversary"], r["split"], r["method"]): r for r in fit_rows}
    adversaries = sorted(df["adversary"].unique())
    fig, axes = plt.subplots(1, len(adversaries), figsize=(6 * len(adversaries), 4.5), squeeze=False)
    for ax, adversary in zip(axes[0], adversaries):
        sub = df[df["adversary"] == adversary]
        for (split, method), grp in sub.groupby(["split", "method"]):
            grp = grp.sort_values("pool_size")
            ax.errorbar(grp["pool_size"], grp["rank1_mean"], yerr=grp["rank1_std"],
                        marker="o", capsize=3, label=f"{split}/{method}")
            fit = fits.get((adversary, split, method))
            # Only extrapolate splits measured to a large enough pool; a 3-point
            # fit (e.g. cross-setup, N<=15) over-extrapolates wildly to N=106.
            if fit and int(fit.get("max_pool_measured", 0)) >= EXTRAPOLATE_MIN_POOL:
                xs = np.linspace(grp["pool_size"].min(), full_pool, 50)
                ax.plot(xs, fit["fit_intercept"] + fit["fit_slope"] * np.log(xs),
                        ls="--", alpha=0.6)
                ax.scatter([full_pool], [fit["extrapolated_rank1_full_pool"]],
                           marker="*", s=140, zorder=5)
        ax.axvline(full_pool, color="gray", ls=":", alpha=0.5)
        ax.set_xlabel("gallery identity-pool size $N$")
        ax.set_ylabel("rank-1 accuracy")
        ax.set_title(f"{adversary} identity leakage vs pool size")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FEATURES_DIR / "identity_pool_scaling.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {out}")


if __name__ == "__main__":
    raise SystemExit(main())
