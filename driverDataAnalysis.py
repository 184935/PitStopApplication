"""
Diagnostic plots for the Driver column (887 anonymised IDs).
Run from the folder that contains Data/train.csv. Figures are saved to Figures/.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE = Path("Data") / "train.csv"
FIG_DIR = Path("Figures")
FIG_DIR.mkdir(exist_ok=True)

LABEL = "PitNextLap"
MIN_ROWS = 50  # drivers with fewer rows are left out of the pit-rate plot

pitstop = pd.read_csv(DATA_FILE)
overall = pitstop[LABEL].mean()


def save(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{name}.png", dpi=150)
    plt.close()


per_driver = pitstop.groupby("Driver").agg(
    rows=(LABEL, "size"),
    pit_rate=(LABEL, "mean"),
    races=("Race", "nunique"),
    years=("Year", "nunique"),
)
per_driver["race_years"] = pitstop.assign(
    ry=pitstop["Race"] + "|" + pitstop["Year"].astype(str)
).groupby("Driver")["ry"].nunique()

print(per_driver["rows"].describe())
print("IDs with >= 1000 rows:", (per_driver["rows"] >= 1000).sum())
print("IDs with < 100 rows:  ", (per_driver["rows"] < 100).sum())
top30_share = per_driver["rows"].sort_values(ascending=False).head(30).sum() / len(pitstop)
print(f"Share of rows from the 30 biggest IDs: {top30_share:.1%}")


# 1. Rows per driver, biggest first: is there a few-drivers-dominate pattern?
sizes = per_driver["rows"].sort_values(ascending=False).values
plt.plot(range(1, len(sizes) + 1), sizes)
plt.yscale("log")
plt.xlabel("Driver ID, ranked by number of rows")
plt.ylabel("Rows (log scale)")
plt.title("Rows per driver ID")
save("driver_01_rows_per_driver")


# 2. Pit rate vs number of rows, with the band expected from pure sampling noise.
#    If drivers did not differ, points would scatter inside the band (a funnel).
d = per_driver[per_driver["rows"] >= MIN_ROWS]
n_grid = np.logspace(np.log10(d["rows"].min()), np.log10(d["rows"].max()), 200)
se = np.sqrt(overall * (1 - overall) / n_grid)
lo = np.clip(overall - 1.96 * se, 0, 1)
hi = np.clip(overall + 1.96 * se, 0, 1)

outside = (d["pit_rate"] < overall - 1.96 * np.sqrt(overall * (1 - overall) / d["rows"])) | \
          (d["pit_rate"] > overall + 1.96 * np.sqrt(overall * (1 - overall) / d["rows"]))

plt.scatter(d.loc[~outside, "rows"], d.loc[~outside, "pit_rate"], s=12, alpha=0.6, label="Inside band")
plt.scatter(d.loc[outside, "rows"], d.loc[outside, "pit_rate"], s=12, alpha=0.8, color="tab:red",
            label="Outside band")
plt.axhline(overall, color="gray", linestyle="--", label="Overall rate")
plt.plot(n_grid, lo, color="gray", linewidth=0.8)
plt.plot(n_grid, hi, color="gray", linewidth=0.8, label="95% band from sampling noise")
plt.xscale("log")
plt.xlabel(f"Rows for the driver ID (log scale, IDs with >= {MIN_ROWS} rows)")
plt.ylabel("Pit rate next lap")
plt.title("Pit rate by driver ID")
plt.legend()
save("driver_02_pit_rate_funnel")
print(f"{outside.mean():.1%} of the {len(d)} IDs fall outside the band")


# 3. How widely does each ID appear? (rows vs distinct race-years)
plt.scatter(per_driver["rows"], per_driver["race_years"], s=12, alpha=0.6)
plt.xscale("log")
plt.xlabel("Rows for the driver ID (log scale)")
plt.ylabel("Distinct race-years the ID appears in")
plt.title("Spread of each driver ID across races")
save("driver_03_spread")

print(f"\nFigures saved in {FIG_DIR.resolve()}")