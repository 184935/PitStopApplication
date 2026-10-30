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
overall_rate = pitstop[LABEL].mean()
#print(pitstop.head())

#print(len(pitstop))




#Plot a histogram of the pitnextlap column
#plt.hist(pitstop[LABEL], bins=2)

# Add axis labels and a title
#plt.xlabel('Pit next lap')
#plt.ylabel('Count')
#plt.title('Bar plot of Pit next lap')

# Alternatively bar plot
counts = pitstop[LABEL].value_counts().sort_index()
pct = counts / counts.sum() * 100

#plt.bar(['No pit (0)', 'Pit (1)'], counts)

# show the plot
#plt.show()



# split features into x, and labels into y
always_drop = ['id', LABEL]
context_cols = ['Driver', 'Race', 'Year']

x_base = pitstop.drop(columns=always_drop + context_cols)
x_full = pitstop.drop(columns=always_drop)
y = pitstop[LABEL]

#print(x_base.head())

# Pit rate by tyre life
g = pitstop.groupby('TyreLife')[LABEL].agg(['mean', 'count'])
g = g[g['count'] >= MIN_N]
plt.plot(g.index, g['mean'])
plt.axhline(overall_rate, color='gray', linestyle='--', label='Overall rate')
plt.xlabel('Tyre life (laps)')
plt.ylabel('Pit rate next lap')
plt.title('Pit rate by tyre life')
plt.legend()
save('pit_rate_by_tyre_life')

# Split by compound
for compound, sub, in pitstop.groupby('Compound'):
    g = sub.groupby('TyreLife')[LABEL].agg(['mean', 'count'])
    g = g[g['count'] >= MIN_N]
    plt.plot(g.index, g['mean'], label=compound)
plt.xlabel('Tyre life (laps)')
plt.ylabel('Pit rate next lap')
plt.title('Pit rate by tyre life and compound')
plt.legend(title='Compound')
save('pit_rate_by_tyrelife_compound')

# Pit rate by RaceProgress
bins = np.linspace(0, 1, 25) # 20 bins
progress_bin = pd.cut(pitstop['RaceProgress'], bins=bins, include_lowest=True)
g = pitstop.groupby(progress_bin, observed=True)[LABEL].mean()
centers = [interval.mid for interval in g.index]
plt.plot(centers, g.values, marker='o')
plt.axhline(overall_rate, color='gray', linestyle='--', label='Overall rate')
plt.xlabel('Race progress (0 = start, 1 = finish)')
plt.ylabel('Pit rate next lap')
plt.title('Pit rate by race progress')
plt.legend()
save('pit_rate_by_raceprogress')

# Monaco interaction test
MONACO_PATTERN = 'Monaco'
MAX_POSITION = 3 # leading positions
MIN_STINT = 2 # at least one stop already made

is_monaco = pitstop['Race'].str.contains(MONACO_PATTERN, case=False, na=False)
is_leader = pitstop['Position'] <= MAX_POSITION
is_later_stint = pitstop['Position'] >= MIN_STINT

groups = {
    'All laps': pd.Series(True, index=pitstop.index),
    f'Stints>={MIN_STINT}, \nP1-{MAX_POSITION}\n(all races)': is_later_stint & is_leader,
    'Monaco\n(all laps)': is_monaco,
    f'Monaco, \nStint>={MIN_STINT},\nP1-{MAX_POSITION}': is_monaco & is_later_stint & is_leader,
}
labels, rates, sizes = [], [], []
for name, mask in groups.items():
    labels.append(name)
    rates.append(pitstop.loc[mask, LABEL].mean() if mask.any() else np.nan)
    sizes.append(int(mask.sum()))

print('\nMonaco test (group, rows, pit rate):')
for name, n, r in zip(labels, sizes, rates):
    print(f' {name.replace(chr(10), " ")}: n={n:,}, rate={r:.3%}')

bars = plt.bar(labels, rates)
for bar, n in zip(bars, sizes):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
             f'n={n:,}', ha='center', va='bottom', fontsize=8)
plt.ylabel('Pit rate next lap')
plt.title('Monaco interaction: leaders after their first stop')
save('monaco_interaction')

# Pit rate by Stint
g = pitstop.groupby("Stint")[LABEL].agg(['mean', 'count'])
g = g[g['count'] >= MIN_N]
plt.bar(g.index.astype(str), g['mean'])
plt.axhline(overall_rate, color='gray', linestyle='--', label='Overall rate')
plt.xlabel('Stint')
plt.ylabel('Pit rate next lap')
plt.title('Pit rate by stint')
plt.legend()
save('pit_rate_by_stint')

# Pit rate by position change
pc = pitstop['Position_Change'].clip(-10, 10).round()
g = pitstop.groupby(pc)[LABEL].agg(['mean', 'count'])
g = g[g['count'] >= MIN_N]
plt.bar(g.index.astype(int).astype(str), g['mean'])
plt.axhline(overall_rate, color='gray', linestyle='--', label= 'Overall rate')
plt.xlabel('Position change (clipped to -10..10)')
plt.ylabel('Pit rate next lap')
plt.legend()
save('pit_rate_by_position_change')

# Distribution of LapTime_delta and cumulative_degradation by class
for col in ['LapTime_Delta', 'Cumulative_Degradation']:
    lo, hi = pitstop[col].quantile([0.01, 0.99]) # clip extreme outliers 
    for value, name in [(0, 'No pit next lap'), (1, 'Pit next lap')]:
        data = pitstop.loc[pitstop[LABEL] == value, col].clip(lo, hi)
        plt.hist(data, bins=50, alpha=0.5, density=True, label=name)
    plt.xlabel(f'{col} (clipped to 1st-99th percentile)')
    plt.ylabel('Density')
    plt.title(f'{col} by class')
    plt.legend()
    save(f'dist_{col}')

# Pit rate by Race
g = pitstop.groupby('Race')[LABEL].agg(['mean', 'count'])
g = g[g['count'] >= MIN_N].sort_values('mean')
plt.figure(figsize=(8, max(4, 0.3 * len(g))))
plt.barh(g.index, g['mean'])
plt.axvline(overall_rate, color='gray', linestyle='--', label='Overall rate')
plt.xlabel('Pit rate next lap')
plt.ylabel('Pit rate next lap')
plt.title('Pit rate by race')
plt.legend()
save('pit_rate_by_race')

# Correlation heatmap
numeric = pitstop.select_dtypes(include='number').drop(columns=['id'], errors='ignore')
corr = numeric.corr()
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha='right')
ax.set_yticklabels(corr.columns)
fig.colorbar(im, label='Correlation')
ax.set_title('Correlation of numeric columns')
save('correlation_heatmap')

# Pit rate by PitStop (this lap)
g = pitstop.groupby('PitStop')[LABEL].agg(['mean', 'count'])
plt.bar(g.index.astype(str), g['mean'])
plt.xlabel('Pitstop (stop on the current lap)')
plt.ylabel('Pit rate next lap')
plt.title('Pit rate next lap by whether the car pitted this lap')
save('pit_rate_by_pitstop')

print('Done')