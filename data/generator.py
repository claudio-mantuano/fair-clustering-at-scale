# MIT License
# Copyright (c) 2026 Claudio Mantuano, University of Bern
# Paper: https://arxiv.org/abs/2605.13759

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

X, y = make_blobs(
    n_samples=21,
    centers=[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
    cluster_std=0.5,
    random_state=13,
)
df = pd.DataFrame(np.round(X, 2), columns=["X1", "X2"])
df["X3"] = y
df.to_csv("data/synthetic_a.csv", index=False)

X, y = make_blobs(
    n_samples=21,
    centers=[[0.1, 0.1], [0.5, 0.9], [0.9, 0.1]],
    cluster_std=[0.17, 0.22, 0.17],
    random_state=3,
)
df = pd.DataFrame(np.round(X, 2), columns=["X1", "X2"])
df["X3"] = y
df.to_csv("data/synthetic_b.csv", index=False)
