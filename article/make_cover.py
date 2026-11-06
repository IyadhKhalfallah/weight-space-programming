"""Cover image for the article (dev.to ratio 1000x420 -> rendered 2000x840)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BG, INK, INK2, MUTED = "#0f1115", "#f2f1ec", "#b9b8b0", "#6c6b66"
ORANGE, BLUE, CELL = "#f07a45", "#4a90e2", "#1d2026"

fig = plt.figure(figsize=(10, 4.2), dpi=200)
fig.patch.set_facecolor(BG)

# right side: a weight matrix, with one column being rewritten
ax = fig.add_axes([0.56, 0.08, 0.40, 0.84]); ax.set_facecolor(BG); ax.axis("off")
rng = np.random.default_rng(3)
R, C = 11, 14
W = rng.normal(0, 1, (R, C))
edit_col = 9
for i in range(R):
    for j in range(C):
        v = np.tanh(W[i, j] / 1.5)
        if j == edit_col:
            col = ORANGE; a = 0.35 + 0.6 * abs(v)
        else:
            col = BLUE if v > 0 else "#8a8f99"; a = 0.10 + 0.45 * abs(v)
        ax.add_patch(FancyBboxPatch((j + 0.1, R - i - 1 + 0.1), 0.8, 0.8, boxstyle="round,pad=0,rounding_size=0.18",
                                    color=col, alpha=a, lw=0))
ax.add_patch(FancyBboxPatch((edit_col - 0.05, -0.15), 1.1, R + 0.3, boxstyle="round,pad=0,rounding_size=0.3",
                            fill=False, ec=ORANGE, lw=1.6))
ax.text(edit_col + 0.5, R + 0.55, "edit", color=ORANGE, ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_xlim(-0.2, C + 0.2); ax.set_ylim(-0.6, R + 1.6); ax.set_aspect("equal")

# left side: text
fig.text(0.055, 0.80, "WEIGHT-SPACE PROGRAMMING", color=ORANGE, fontsize=11, fontweight="bold")
fig.text(0.055, 0.72, "Editing models\ninstead of training them", color=INK, fontsize=25, fontweight="bold", va="top", linespacing=1.15)
fig.text(0.055, 0.37, "Tested on 6 kinds of models:\nwhat works, and where it breaks.", color=INK2, fontsize=12.5, va="top", linespacing=1.4)
fig.text(0.055, 0.12, "0 labels → AUC 0.809", color=INK, fontsize=12, fontweight="bold")
fig.text(0.268, 0.12, "vs 0.839 trained on 2.5 years", color=MUTED, fontsize=12)

fig.savefig("article/img/0_cover.png", facecolor=BG)
print("ok")
