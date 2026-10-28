"""Bar chart of field balanced accuracy per training condition, from results.json."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(here, "results.json")))
rows = [("Lab photos only", r["lab"], "#2a6fb0"),
        ("+ standard augmentation", r["lab+augment"], "#2a6fb0"),
        ("+ diffusion v1 (wide scene)", r["lab+diffusion v1"], "#2a6fb0"),
        ("+ diffusion v2 (close-up)", r["lab+diffusion v2"], "#2a6fb0"),
        ("Real field photos (reference)", r["field photos, 5-fold (reference)"], "#9e9e9e")]
fig, ax = plt.subplots(figsize=(7, 3.6))
for i, (name, v, color) in enumerate(rows):
    lo, hi = v["ci95"]
    ax.barh(i, v["balanced_accuracy"], color=color, height=0.6)
    ax.plot([lo, hi], [i, i], color="#333333", lw=1.5)
    ax.text(hi + 0.01, i, f"{v['balanced_accuracy']:.0%}", va="center", fontsize=9, color="#333333")
ax.axvline(1 / 3, color="#777777", ls="--", lw=1)
ax.text(1 / 3 + 0.005, -0.5, "chance", fontsize=8, color="#777777")
ax.set_yticks(range(len(rows)), [n for n, _, _ in rows], fontsize=9)
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_xlabel("Balanced accuracy on 375 field photos (95% bootstrap CI)", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title(f"Maize disease classifier trained on lab photos ({r['lab photos, tested on held-out lab photos (5-fold)']['balanced_accuracy']:.0%} on lab photos)",
             fontsize=10, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(here, "lab_to_field_chart.png"), dpi=150)
print("saved")
