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

pitstop = pd.read_csv('Data/train.csv')

pitstop["RaceYear"] = pitstop["Race"] + "|" + pitstop["Year"].astype(str)
pitstop.to_csv("Data/train_prepped.csv", index=False);