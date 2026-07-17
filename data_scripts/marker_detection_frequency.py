import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


def load_poses(path):
    p = pd.read_csv(os.path.expanduser(path))
    p = p.dropna(subset=["marker_id"]).copy()
    p["marker_id"] = p["marker_id"].astype(int)
    return p


def capture_rates(poses, expected_ids=None):
    """
    Percentage of frames each marker was detected in.
    Denominator = full frame span (max-min+1), so frames where nothing was
    detected still count against every marker. Generalizes to any marker set.
    """
    frames = poses["frame"]
    total = int(frames.max() - frames.min() + 1)

    ids = sorted(poses["marker_id"].unique())
    if expected_ids:
        ids = sorted(set(ids) | set(expected_ids))

    rows = []
    for mid in ids:
        seen = poses.loc[poses.marker_id == mid, "frame"].nunique()
        rows.append((mid, seen, 100.0 * seen / total))
    return total, pd.DataFrame(rows, columns=["marker_id", "frames_seen", "pct"])


def capture_rate_plot(poses, expected_ids=None, save=None):
    total, df = capture_rates(poses, expected_ids)
    n = len(df)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n))

    fig, ax = plt.subplots(figsize=(max(6, 1.1 * n + 2), 5))
    bars = ax.bar([str(m) for m in df.marker_id], df.pct, color=colors)
    for b, pct, seen in zip(bars, df.pct, df.frames_seen):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6,
                f"{pct:.1f}%\n{seen}/{total}", ha="center", va="bottom",
                fontsize=8)

    ax.set_xlabel("marker ID")
    ax.set_ylabel("frames captured [%]")
    ax.set_title(f"Marker capture rate  (total frames = {total})")
    ax.set_ylim(0, min(100, df.pct.max() * 1.25 + 5))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=110)
    return total, df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Per-marker capture rate (%% of frames detected).")
    ap.add_argument("poses", nargs="?",
                    default="~/TARES_SITL/poses_140mm.csv")
    ap.add_argument("--expected-ids", type=int, nargs="*", default=None,
                    help="marker IDs that should exist; missing ones show as 0%%")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    poses = load_poses(args.poses)
    total, df = capture_rate_plot(
        poses,
        expected_ids=args.expected_ids,
        save=(os.path.join(os.path.expanduser(args.outdir),
              os.path.splitext(os.path.basename(args.poses))[0] + "_capture_rate.png")
              if args.outdir else None))

    print(f"total frames: {total}")
    for _, r in df.iterrows():
        print(f"  marker {int(r.marker_id):>4}: {r.pct:5.1f}%  ({int(r.frames_seen)}/{total})")

    if not args.outdir:
        plt.show()
