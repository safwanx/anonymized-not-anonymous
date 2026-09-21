#!/usr/bin/env python3
# Combine saved evaluation outputs and write the final summary/plot.

from pipeline_common import *


def read_results(filename: str) -> pd.DataFrame:
    path = FEATURES_DIR / filename
    if path.exists():
        return pd.read_csv(path)
    print(f"Missing results file: {path}")
    return pd.DataFrame()


df_face = read_results("face_recognition_results.csv")
df_reid = read_results("person_reid_results.csv")
df_sil = read_results("silhouette_proxy_results.csv")
df_pose = read_results("pose_identity_results.csv")
df_utility = read_results("action_utility_proxy_results.csv")

privacy_frames = [df for df in [df_face, df_reid, df_sil] if not df.empty]
df_privacy = pd.concat(privacy_frames, ignore_index=True) if privacy_frames else pd.DataFrame()
df_privacy.to_csv(FEATURES_DIR / "combined_privacy_attack_results.csv", index=False)

print("=" * 80)
print("PRIVACY ATTACK SUMMARY")
print("=" * 80)
if not df_privacy.empty:
    summary_cols = [
        "attack",
        "protocol",
        "method",
        "rank1",
        "rank5",
        "mAP",
        "probe_coverage",
        "coverage_adjusted_rank1",
    ]
    existing = [col for col in summary_cols if col in df_privacy.columns]
    print(df_privacy[existing].to_string(index=False, float_format="{:.3f}".format))
else:
    print("No privacy attack results were generated.")

print("\nPOSE IDENTITY SUMMARY")
if not df_pose.empty:
    print(df_pose.to_string(index=False, float_format="{:.3f}".format))
else:
    print("No pose results were generated.")

print("\nACTION UTILITY PROXY SUMMARY")
if not df_utility.empty:
    print(df_utility.to_string(index=False, float_format="{:.3f}".format))
else:
    print("No utility proxy results were generated.")

if not df_privacy.empty:
    plot_df = df_privacy.copy()
    plot_df["label"] = plot_df["attack"] + " / " + plot_df["protocol"]
    labels = sorted(plot_df["label"].unique())
    fig, axes = plt.subplots(len(labels), 1, figsize=(12, max(4, 2.8 * len(labels))))
    axes = np.atleast_1d(axes)
    for ax, label in zip(axes, labels):
        part = plot_df[plot_df["label"] == label].sort_values("method")
        ax.barh(part["method"], part["coverage_adjusted_rank1"])
        ax.set_xlim(0, 1)
        ax.set_title(label)
        ax.set_xlabel("Coverage-adjusted Rank-1")
    plt.tight_layout()
    plt.savefig(str(OUTPUT_ROOT / "privacy_attack_results.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

print(f"Outputs written under: {OUTPUT_ROOT}")
