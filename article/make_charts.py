"""Charts for the article. Rain charts are recomputed from the data; the others use the measured results in README.md."""
import json, warnings, os
import numpy as np, xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from wsl.workspace import load_csv
from wsl.xgb_write import compile_program
warnings.filterwarnings("ignore")
OUT = "article/img"; os.makedirs(OUT, exist_ok=True)

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False})


def title(fig, t, sub):
    fig.text(0.04, 0.95, t, fontsize=19, fontweight="bold", color=INK, va="top")
    fig.text(0.04, 0.885, sub, fontsize=12.5, color=INK2, va="top")


# ---------- rain: recompute ----------
d = load_csv("experiments/e7/weather.csv")
F = ["month", "lat", "pressure", "pressure_change_24h", "pressure_change_6h", "humidity", "humidity_evening", "cloud",
     "cloud_evening", "wind_speed_evening", "wind_from_west", "wind_from_south", "temp", "dewpoint_spread_evening", "rain_today", "rain_evening"]
X = np.c_[[d[f] for f in F]].T; y = d["rain_tomorrow"].astype(int)
tr, te = d["date"] < "2026-07-01", d["date"] >= "2026-07-01"
dm = xgb.DMatrix(X[te], feature_names=F)
s1 = compile_program(json.load(open("experiments/e7/rain_program.json")), F, X[tr], "/tmp/_r1.json").predict(dm, output_margin=True)
s3 = compile_program(json.load(open("experiments/e7/rain_program_v3.json")), F, X[tr], "/tmp/_r3.json").predict(dm, output_margin=True)
full = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=.05, subsample=.8).fit(X[tr], y[tr]).predict(X[te], output_margin=True)
rng = np.random.default_rng(0); few = rng.choice(np.where(tr)[0], 60, replace=False)
small = xgb.XGBClassifier(n_estimators=100, max_depth=2, learning_rate=.1).fit(X[few], y[few]).predict(X[te], output_margin=True)
pers = d["rain_today"][te] - 1.0
auc = lambda s: roc_auc_score(y[te], s)

# 1. hero bar chart
rows = [("Yesterday's weather\n(persistence)", auc(pers), GRAY, "no model"),
        ("XGBoost trained\non 60 days", auc(small), GRAY, "60 labels"),
        ("Written by AI\nv1", auc(s1), ORANGE, "0 labels"),
        ("Written by AI, v3\n(revised from 2024–25 data)", auc(s3), ORANGE, "read old labels, no training"),
        ("XGBoost trained\non 2.5 years", auc(full), BLUE, "9,110 labels")]
fig, ax = plt.subplots(figsize=(11, 6.2)); fig.subplots_adjust(left=0.29, right=0.95, top=0.76, bottom=0.12)
yy = np.arange(len(rows))[::-1]
for yi, (n, v, c, note) in zip(yy, rows):
    ax.barh(yi, v - 0.5, left=0.5, height=0.56, color=c)
    ax.text(v + 0.004, yi, f"{v:.3f}", va="center", fontsize=14, fontweight="bold", color=INK)
    ax.text(v + 0.045, yi, note, va="center", fontsize=11.5, color=MUTED)
ax.set_yticks(yy, [r[0] for r in rows], fontsize=12.5, color=INK)
ax.set_xlim(0.5, 0.97); ax.set_xlabel("AUC on Jul–Oct 2026 (1,220 city-days, after the AI's knowledge cutoff)", fontsize=11.5)
ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
title(fig, "An AI wrote a rain forecaster with zero training labels",
      "Predicting rain tomorrow in 10 European cities. Orange = hand-written trees, compiled into a real XGBoost file.")
fig.savefig(f"{OUT}/1_rain_hero.png", dpi=160); plt.close(fig)

# 2. ROC curves
fig, ax = plt.subplots(figsize=(8, 7.4)); fig.subplots_adjust(left=0.12, right=0.95, top=0.80, bottom=0.11)
for s, c, n, ls in [(full, BLUE, "Trained on 2.5 years", "-"), (s1, ORANGE, "Written by AI, 0 labels", "-"), (pers, GRAY, "Persistence", "--")]:
    fpr, tpr, _ = roc_curve(y[te], s); ax.plot(fpr, tpr, color=c, lw=2.4, ls=ls, label=f"{n}  (AUC {auc(s):.3f})")
ax.plot([0, 1], [0, 1], color=GRID, lw=1.2)
ax.set_xlabel("False alarms (dry days predicted rainy)"); ax.set_ylabel("Rain days caught")
ax.legend(frameon=False, loc="lower right", fontsize=12); ax.grid(True, color=GRID); ax.set_axisbelow(True)
title(fig, "Close to the trained model at every threshold", "ROC curves on the unseen test period")
fig.savefig(f"{OUT}/2_rain_roc.png", dpi=160); plt.close(fig)

# 3. fixes without retraining: before/after small multiples (measured, from README)
fixes = [("Fraud model: Brazilian\ncustomers falsely flagged", "%", 7.2, 2.0, "lower is better", "32 weights"),
         ("Fraud model: gift-card\nscam caught", "%", 10, 98, "higher is better", "1 rule, 23 weights"),
         ("Ticket router: account-takeover\ntickets routed right", "%", 2.6, 95.8, "higher is better", "1 rule, 31 weights"),
         ("Delivery-time model:\nerror (MAE, minutes)", " min", 5.7, 3.2, "lower is better", "2 rules, 42 weights"),
         ("Game agent fooled by a\nrendering bug: score", "", -0.1, 19.3, "higher is better", "hand-written circuit"),
         ("Logistic regression:\nBrazil false flags", "%", 6.8, 2.0, "lower is better", "1 coefficient")]
fig, axs = plt.subplots(2, 3, figsize=(13, 8.2)); fig.subplots_adjust(left=0.04, right=0.98, top=0.78, bottom=0.05, hspace=0.75, wspace=0.35)
for ax, (n, u, b, a, better, cost) in zip(axs.flat, fixes):
    ax.bar([0, 1], [b, a], width=0.6, color=[GRAY, AQUA if "higher" in better else BLUE])
    top = max(b, a)
    for x, v in [(0, b), (1, a)]:
        ax.text(x, max(v, 0) + top * 0.04, f"{v:g}{u}", ha="center", fontsize=14, fontweight="bold", color=INK)
    ax.set_ylim(min(0, b, a) * 1.3, top * 1.28); ax.set_xticks([0, 1], ["before", "after edit"], fontsize=12, color=INK)
    ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.axhline(0, color=GRID)
    ax.set_title(n, fontsize=12.5, color=INK, loc="left", pad=8)
    ax.text(0, -0.30, f"{better} · {cost}", transform=ax.transAxes, fontsize=10.5, color=MUTED)
title(fig, "Fixing trained models by editing weights, no retraining",
      "Each fix: diagnose inside the model, write a small edit, test before/after on new data. Edits are reversible.")
fig.savefig(f"{OUT}/3_fixes.png", dpi=160); plt.close(fig)

# 4. verdict grid
cols = ["Diagnose", "Fix by editing", "Write from scratch"]
rowsv = [("Logistic regression", [2, 2, -1]), ("Small neural nets (MLP)", [2, 2, 2]), ("Trees / XGBoost", [2, 2, 2]),
         ("RL policy (game agent)", [2, 2, 2]), ("CNN (images)", [2, 1, 1]), ("LLM (0.8B, Qwen)", [2, 0, -1])]
lab = {2: "works", 1: "partial", 0: "didn't generalize", -1: "not tried"}
col = {2: "#d6efe4", 1: "#fbe6c7", 0: "#f6d4d1", -1: "#efeeea"}
fig, ax = plt.subplots(figsize=(11, 6.4)); fig.subplots_adjust(left=0.03, right=0.97, top=0.80, bottom=0.10)
ax.set_xlim(0, 4); ax.set_ylim(0, len(rowsv) + 1); ax.axis("off")
for j, c in enumerate(cols): ax.text(1.5 + j * 0.85, len(rowsv) + 0.5, c, ha="center", va="center", fontsize=13, fontweight="bold", color=INK)
for i, (n, vals) in enumerate(rowsv):
    yv = len(rowsv) - 1 - i
    ax.text(0.05, yv + 0.5, n, va="center", fontsize=13, color=INK)
    for j, v in enumerate(vals):
        ax.add_patch(plt.Rectangle((1.1 + j * 0.85, yv + 0.08), 0.8, 0.84, color=col[v], lw=0))
        ax.text(1.5 + j * 0.85, yv + 0.5, lab[v], ha="center", va="center", fontsize=12, color=INK)
fig.text(0.04, 0.05, "CNN: the shortcut was removed, but a skill it never learned was only partly hand-written (52% → 71% accuracy). "
         "LLM: causes found, but my edit did not reproduce the published effect.", fontsize=9.8, color=MUTED, wrap=True)
title(fig, "Where it works today", "My results across model types. Reading a model works everywhere; editing big ones is the open problem.")
fig.savefig(f"{OUT}/4_verdict.png", dpi=160); plt.close(fig)

# 5. Qwen stop probability
vals = [("Original Qwen3.5-0.8B", 1.09e-5, GRAY), ("Published edit\n(from the paper)", 3.12e-4, BLUE), ("My edit\n(found by search)", 1.28e-5, ORANGE)]
fig, ax = plt.subplots(figsize=(10, 5.6)); fig.subplots_adjust(left=0.27, right=0.93, top=0.70, bottom=0.14)
for yi, (n, v, c) in zip([2, 1, 0], vals):
    r = v / vals[0][1]
    ax.barh(yi, r, height=0.55, color=c); ax.text(r + 0.4, yi, f"{r:.1f}×", va="center", fontsize=14, fontweight="bold", color=INK)
ax.set_yticks([2, 1, 0], [v[0] for v in vals], fontsize=12.5, color=INK); ax.tick_params(axis="y", length=0)
ax.set_xlabel("Probability of ending its reasoning (</think>), relative to the original model"); ax.set_xlim(0, 33)
ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
title(fig, "On an LLM, I found the brake but couldn't press it",
      "The published edit makes the model ~29× more likely to stop overthinking.\nMine, found by automated search, barely moved it.")
fig.savefig(f"{OUT}/5_llm.png", dpi=160); plt.close(fig)
print("ok", {n: round(v, 3) for n, v, *_ in rows})
