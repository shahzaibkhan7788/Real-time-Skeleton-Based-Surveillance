"""
Generate training/validation diagnostics for SPARTA-C and SPARTA-F runs.

The script reads the logged metrics from:
  - results-sparta-c/metrics.csv
  - results-sparta-f/metrics.csv

Outputs a folder structure like:
  visualizations/
    sparta-c/
      loss.png
      auc.png
      eer_threshold.png
      lr_vs_eval_loss.png
      fpr_at_target_fnr.png
      threshold_stability.png
      loss_vs_auc.png
    sparta-f/
      ... (same filenames)

Run from the project root:
    python visulation.py

Notes
-----
* Any malformed lines (e.g., CLI args appended to metrics.csv) are skipped.
* "final_best" rows are used for reference lines where available.
"""

# python -c "import numpy as np; data=np.load('/home/waleed64/Documents/Human_Centric_Anomaly_Detection_Agent/Preprocessedd-data/labels/01_0209.npy'); print(data)"
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


# Use a pleasant grid style; fall back gracefully if seaborn styles are absent.
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        plt.style.use("ggplot")

DATASETS: Dict[str, Path] = {
    "sparta-c": Path("results-sparta-c/metrics.csv"),
    "sparta-f": Path("results-sparta-f/metrics.csv"),
}


def load_metrics(csv_path: Path) -> Tuple[pd.DataFrame, Optional[Dict[str, float]]]:
    """Load metrics, skipping malformed lines and separating the final_best row."""
    df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")

    # Keep numeric epochs for plotting
    df["epoch_num"] = pd.to_numeric(df["epoch"], errors="coerce")
    metric_rows = df.dropna(subset=["epoch_num"]).copy()
    metric_rows["epoch_num"] = metric_rows["epoch_num"].astype(int)

    # Extract the final_best summary if present
    best_mask = df["epoch"].astype(str).str.contains("final_best", case=False, na=False)
    best_row = df.loc[best_mask].iloc[0].to_dict() if best_mask.any() else None

    # Convert numeric-like fields in best_row
    if best_row:
        for key, val in list(best_row.items()):
            if key == "epoch":
                continue
            try:
                best_row[key] = float(val)
            except (TypeError, ValueError):
                best_row[key] = None

    return metric_rows, best_row


def annotate_min(ax, df: pd.DataFrame, column: str, color: str, label: str) -> int:
    """Mark the minimum point of a column and return its epoch."""
    idx = df[column].idxmin()
    epoch = int(df.loc[idx, "epoch_num"])
    value = df.loc[idx, column]
    ax.scatter(epoch, value, color=color, zorder=5, s=45)
    ax.annotate(
        f"{label}\nmin @ {epoch}",
        xy=(epoch, value),
        xytext=(epoch + 1, value * 1.05),
        arrowprops=dict(arrowstyle="->", color=color, lw=1),
        fontsize=8,
        color=color,
    )
    return epoch


def savefig(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_loss(df: pd.DataFrame, best: Optional[Dict[str, float]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(df["epoch_num"], df["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    ax.plot(df["epoch_num"], df["eval_loss_mean"], label="Eval Loss", color="#d62728", lw=2)

    # Highlight the post-minimum region (possible overfitting window)
    min_epoch = annotate_min(ax, df, "eval_loss_mean", color="#d62728", label="Eval loss")
    ax.axvspan(min_epoch, df["epoch_num"].max(), color="#d62728", alpha=0.06, label="After eval-loss min")

    ax.set_title("Training vs Evaluation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.4)

    if best and best.get("eval_loss_mean") is not None:
        ax.axhline(best["eval_loss_mean"], color="#d62728", ls=":", lw=1, label="Best eval loss")

    savefig(fig, out_dir / "loss.png")


def plot_auc(df: pd.DataFrame, best: Optional[Dict[str, float]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(df["epoch_num"], df["auc_roc"], label="AUC-ROC", color="#9467bd", lw=2)
    ax.plot(df["epoch_num"], df["auc_pr"], label="AUC-PR", color="#2ca02c", lw=2)

    if best:
        if best.get("auc_roc") is not None:
            ax.axhline(best["auc_roc"], color="#9467bd", ls=":", lw=1)
        if best.get("auc_pr") is not None:
            ax.axhline(best["auc_pr"], color="#2ca02c", ls=":", lw=1)

    ax.set_title("Discrimination Power")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0.55, 1.0)
    ax.legend()
    ax.grid(True, ls="--", alpha=0.4)

    savefig(fig, out_dir / "auc.png")


def plot_eer(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 4.4))
    ax1.plot(df["epoch_num"], df["eer"], color="#ff7f0e", lw=2, label="EER")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("EER", color="#ff7f0e")
    ax1.tick_params(axis="y", labelcolor="#ff7f0e")

    ax2 = ax1.twinx()
    ax2.plot(df["epoch_num"], df["eer_th"], color="#1f77b4", lw=2, ls="--", label="EER Threshold")
    ax2.set_ylabel("Threshold", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")

    fig.suptitle("Equal Error Rate & Threshold")
    fig.legend(loc="upper right", bbox_to_anchor=(0.92, 0.92))
    ax1.grid(True, ls="--", alpha=0.4)

    savefig(fig, out_dir / "eer_threshold.png")


def plot_lr_and_eval_loss(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 4.4))
    ax1.plot(df["epoch_num"], df["lr"], color="#17becf", lw=2, label="Learning rate")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("LR", color="#17becf")
    ax1.tick_params(axis="y", labelcolor="#17becf")

    ax2 = ax1.twinx()
    ax2.plot(df["epoch_num"], df["eval_loss_mean"], color="#d62728", lw=2, ls="--", label="Eval loss")
    ax2.set_ylabel("Eval loss", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    fig.suptitle("Learning Rate Schedule vs Eval Loss")
    fig.legend(loc="upper right", bbox_to_anchor=(0.92, 0.9))
    ax1.grid(True, ls="--", alpha=0.4)

    savefig(fig, out_dir / "lr_vs_eval_loss.png")


def plot_fpr(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(
        df["epoch_num"],
        df["fpr_at_target_fnr"],
        color="#8c564b",
        lw=2,
        label="FPR @ target FNR",
    )

    ax.set_title("False Positive Rate at Target FNR")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("FPR")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.4)

    savefig(fig, out_dir / "fpr_at_target_fnr.png")


def plot_threshold_stability(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(df["epoch_num"], df["threshold_at_target_fnr"], color="#7f7f7f", lw=2)
    ax.set_title("Decision Threshold Stability (Target FNR)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Threshold")
    ax.grid(True, ls="--", alpha=0.4)

    savefig(fig, out_dir / "threshold_stability.png")


def plot_loss_vs_auc(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    scatter = ax.scatter(
        df["eval_loss_mean"],
        df["auc_roc"],
        c=df["epoch_num"],
        cmap="viridis",
        s=40,
        edgecolor="k",
        linewidth=0.4,
    )
    ax.set_title("Eval Loss vs AUC-ROC (color = epoch)")
    ax.set_xlabel("Eval loss")
    ax.set_ylabel("AUC-ROC")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Epoch")
    ax.grid(True, ls="--", alpha=0.4)

    savefig(fig, out_dir / "loss_vs_auc.png")


def generate_plots(name: str, csv_path: Path, out_root: Path) -> None:
    df, best = load_metrics(csv_path)
    out_dir = out_root / name

    plot_loss(df, best, out_dir)
    plot_auc(df, best, out_dir)
    plot_eer(df, out_dir)
    plot_lr_and_eval_loss(df, out_dir)
    plot_fpr(df, out_dir)
    plot_threshold_stability(df, out_dir)
    plot_loss_vs_auc(df, out_dir)


def main() -> None:
    out_root = Path("visualizations")
    for name, csv_path in DATASETS.items():
        if not csv_path.exists():
            print(f"[WARN] {csv_path} not found; skipping {name}.")
            continue
        generate_plots(name, csv_path, out_root)
        print(f"[OK] Saved plots for {name} to {out_root / name}")


if __name__ == "__main__":
    main()
