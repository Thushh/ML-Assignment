"""Exploratory figures that motivate the problem and the features.

Run:  python -m src.eda
"""
import numpy as np
import pandas as pd

from .config import CLASS_COLORS, CLASS_NAMES, COLORS, PROCESSED_DIR, SOURCE_FILE, SUMMARY_FILE
from .features import MONTH_ABBR
from .plotting import plt, save


def national_mix():
    src = pd.read_csv(SOURCE_FILE)
    src["date"] = pd.to_datetime(src.Year.astype(str) + "-" + src.Month.map(MONTH_ABBR).astype(str) + "-01")
    group = np.select(
        [src.Source.eq("Major Hydro"), src.Source.eq("Thermal Coal"), src.Source.eq("Thermal Oil")],
        ["Major hydro", "Coal", "Oil"], "NCRE renewables")
    piv = src.assign(group=group).pivot_table(index="date", columns="group",
                                              values="Generation_GWh", aggfunc="sum")
    order = ["Major hydro", "NCRE renewables", "Coal", "Oil"]
    colors = [COLORS["blue"], COLORS["aqua"], "#52514e", COLORS["orange"]]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for col, c in zip(order, colors):
        ax.plot(piv.index, piv[col], color=c, label=col)
        ax.annotate(col, (piv.index[-1], piv[col].iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=COLORS["ink2"])
    ax.set_title("Sri Lanka monthly generation by source, 2023-2024 (GWh)")
    ax.set_ylabel("GWh")
    ax.legend(ncol=4, loc="upper left")
    ax.set_xlim(piv.index[0], piv.index[-1] + pd.Timedelta(days=75))
    save(fig, "eda_national_generation_mix.png")


def cf_heatmap(df):
    piv = df.groupby(["category", "month"]).cf.median().unstack()
    fig, ax = plt.subplots(figsize=(10, 3.6))
    im = ax.imshow(piv.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(12), [m for m in MONTH_ABBR])
    ax.set_yticks(range(len(piv)), piv.index)
    ax.grid(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 0.55 else COLORS["ink"])
    ax.set_title("Median capacity factor by technology and month (monsoon signal)")
    fig.colorbar(im, ax=ax, fraction=0.025, label="Capacity factor")
    save(fig, "eda_cf_heatmap.png")


def class_by_category(df):
    ct = pd.crosstab(df.category, df.cf_class, normalize="index").reindex(columns=[0, 1, 2], fill_value=0)
    counts = df.category.value_counts()
    ct = ct.loc[counts.index[::-1]]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    left = np.zeros(len(ct))
    for k in range(3):
        ax.barh(ct.index, ct[k], left=left, color=CLASS_COLORS[k], label=CLASS_NAMES[k],
                edgecolor=COLORS["surface"], linewidth=2, height=0.6)
        left += ct[k].values
    ax.set_yticks(range(len(ct)), [f"{c}  (n={counts[c]})" for c in ct.index])
    ax.set_xlabel("Share of plant-months")
    ax.set_xlim(0, 1)
    ax.set_title("Target class mix by technology")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.08))
    save(fig, "eda_class_by_category.png")


def renewable_share():
    s = pd.read_csv(SUMMARY_FILE)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(s.Year.astype(str), s.Renewable_Share_Pct, color=COLORS["aqua"], width=0.5)
    for x, v in zip(s.Year.astype(str), s.Renewable_Share_Pct):
        ax.text(x, v + 1, f"{v:.1f}%", ha="center", color=COLORS["ink"])
    ax.axhline(70, color=COLORS["ink2"], linestyle="--", linewidth=1)
    ax.text(-0.4, 71.5, "2030 target: 70 %", color=COLORS["ink2"], fontsize=9)
    ax.set_ylim(0, 80)
    ax.set_title("Renewable share of national generation")
    save(fig, "eda_renewable_share.png")


def main():
    df = pd.read_csv(PROCESSED_DIR / "model_dataset.csv")
    national_mix()
    cf_heatmap(df)
    class_by_category(df)
    renewable_share()
    print("EDA figures written to reports/figures/")


if __name__ == "__main__":
    main()
