import pandas as pd
import numpy as np
import scipy
import open3d as o3d
from tqdm import tqdm
import os
from typing import List, Dict,Tuple,Any
import catboost
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split,KFold,GroupKFold,StratifiedGroupKFold,StratifiedKFold
import plotly.graph_objects as go
from plotly.subplots import make_subplots

df_mapping = pd.read_csv("filtering_mapping.txt", header=None)
row = df_mapping[df_mapping[0].str.contains('removed from the data set')].index

df_mapping = df_mapping.iloc[:row[0]-1]
df_mapping = df_mapping[~df_mapping[0].str.contains('Remapped labeled:')]
df_mapping[['label', 'name']] = df_mapping[0].str.split(' -- ', expand=True)
df_mapping = df_mapping.drop(0, axis=1)
df_mapping["label"] = df_mapping["label"].str.strip()

df_mapping.head()
df_mapping.index = range(len(df_mapping))