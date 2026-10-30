from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

FIG_DIR = Path("Figures")

CHOSEN = 0.3

RESULTS = {
    0.5: (17156, 7038, 8261, 58150),
    0.4: (19487, 9814, 5930, 55374),
    0.3: (21137, 12712, 4280, 52476),
    0.2: (22490, 15965, 2927, 49223),
}
def draw(ax, tp, fp, fn, tn, threshold):
    cm = np.array([[tn, fp],[fn, tp]])
    row_share = cm / cm.sum(axis=1, keepdims=True)

    ax.imshow(row_share, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            color = "white" if row_share[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]:,}\n({row_share[i,j]:.1%})",
                    ha="center", va="center", color=color, fontsize=11)

    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    ax.set_xticks([0,1])
    ax.set_xticklabels(["No pit", "Pit"])
    ax.set_yticks([0,1])
    ax.set_yticklabels(["No pit", "Pit"])
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Actual")
    ax.set_title(f"Threshold {threshold}\nprecision {precision:3f}, recall {recall:3f}")

# figure 1: chosen threshold
tp, fp, fn, tn = RESULTS[CHOSEN]
fig, ax = plt.subplots(figsize=(5, 4.5))
draw(ax, tp, fp, fn, tn, CHOSEN)
plt.tight_layout()
plt.savefig(FIG_DIR / f"Conusion_matrix_t{int(CHOSEN*100):02d}.png", dpi=150)
plt.close()

# Figure 2: all threshols side by side
fig, axes = plt.subplots(2, 2, figsize=(9,8))
for ax, (t, counts) in zip(axes.ravel(), sorted(RESULTS.items(), reverse=True)):
    draw(ax, *counts, t)
plt.suptitle("Confusion matrices on the held-out test set", fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / "confusion_matrices_all_thresholds.png", dpi=150)
plt.close()