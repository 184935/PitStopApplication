
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import sklearn
import urllib.request
from sklearn.model_selection import train_test_split

NB_DIR = Path.cwd()
DATA = NB_DIR/'Data'
FILE = DATA/'train.csv'
FIG_DIR = Path('Figures')
FIG_DIR.mkdir(exist_ok=True)

def save(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / f'{name}.png', dpi=150)
    plt.close()

LABEL = 'PitNextLap'
MIN_N = 200 # ignore groups with fewer than this to avoid rates getting too noisy.

pitstop = pd.read_csv('Data/train.csv')

zero = pitstop["LapTime_Delta"] == 0
print("Rows with delta == 0:", zero.sum(), f"({zero.mean():.1%})")
print("Pit rate when delta == 0:", pitstop.loc[zero, "PitNextLap"].mean())
print("Pit rate when delta != 0:", pitstop.loc[~zero, "PitNextLap"].mean())

# Does it line up with your "no previous lap" idea?
print(pitstop.loc[zero, "TyreLife"].describe())
print((pitstop.loc[zero, "LapNumber"] == 1).mean())

monaco = pitstop[pitstop["Race"].str.contains("Monaco", case=False, na=False)]
print(monaco.groupby("Year")["PitNextLap"].agg(["mean", "count"]))
print(monaco["Stint"].value_counts().sort_index())

print(pitstop.groupby("Stint")[LABEL].agg(["mean", "count"]))
print(pitstop.groupby("Stint")["TyreLife"].describe())
print(pitstop.groupby("Stint")["RaceProgress"].mean())

print(pitstop.groupby("Stint")["RaceProgress"].describe())
print(pitstop["Stint"].value_counts().sort_index())

bins = np.linspace(0, 1, 11)
pitstop["ProgressBin"] = pd.cut(pitstop["RaceProgress"], bins=bins, include_lowest=True)

print(pitstop["Compound"].value_counts())

print(pd.crosstab(pitstop["Stint"], pitstop["Compound"], normalize="index").round(2))
print(pd.crosstab(pd.cut(pitstop["RaceProgress"], 5), pitstop["Compound"], normalize="index").round(2))
print(pitstop["Compound"].value_counts())

print(type(pitstop["TyreLife"]))