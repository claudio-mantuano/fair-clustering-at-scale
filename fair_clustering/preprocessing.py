# MIT License
# Copyright (c) 2026 Claudio Mantuano, University of Bern
# Paper: https://arxiv.org/abs/2605.13759

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sklearn import preprocessing
from ucimlrepo import fetch_ucirepo


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def create_batches(
    X: np.ndarray,
    sensitive_features: np.ndarray,
    n_batches: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """
    Construct batches for S-MPFC by clustering the dataset.

    Parameters
    ----------
    X : np.ndarray
        Non-sensitive feature matrix, shape (n_objects, n_features).
    sensitive_features : np.ndarray
        Protected group labels of each sensitive feature, shape
        (n_objects, n_sensitive_features), or shape (n_objects,) for a single
        sensitive feature.
    n_batches : int
        Number of batches (representatives).
    seed : int, default=42
        Random seed for FAISS k-means.

    Returns
    -------
    batch_X : np.ndarray
        Batch centers or representatives, shape (n_batches, n_features).
    batch_map : np.ndarray
        Map from objects to batch labels, shape (n_objects,).
    batch_weights : list[np.ndarray]
        Number of objects per protected group in each batch, one array of shape
        (n_batches, n_groups) per sensitive feature.
    """
    X = X.astype(np.float32)
    n_features = X.shape[1]
    # A single sensitive feature can also be passed as a 1D array
    if sensitive_features.ndim == 1:
        sensitive_features = sensitive_features.reshape(-1, 1)

    model = faiss.Kmeans(d=n_features, k=n_batches, niter=10, seed=seed)
    model.train(X)
    _, labels = model.index.search(X, 1)
    labels = labels.flatten()

    weights = []
    for m in range(sensitive_features.shape[1]):
        n_groups = int(sensitive_features[:, m].max()) + 1
        feature_weights = np.zeros((n_batches, n_groups), dtype=int)
        np.add.at(feature_weights, (labels, sensitive_features[:, m]), 1)
        weights.append(feature_weights)

    return model.centroids, labels, weights


@dataclass
class DataPreprocessor:
    """
    Dataset preprocessing pipeline for fair clustering experiments.

    Parameters
    ----------
    dataset : str
        Dataset name.
    sensitive_names : list[str]
        Names of the sensitive features.
    binary : bool
        If True, keep only the two largest protected groups of each sensitive feature.
    n_subsample : int | None
        Number of objects to subsample, None to keep entire dataset.
    n_batches : int | None
        Number of batches for S-MPFC, None to avoid batch generation.
    n_features : int | None
        Number of features to subsample, None to keep all features.
    normalize : bool
        Apply min-max scaling to non-sensitive features.
    standardize : bool
        Apply standardization to non-sensitive features.

    Attributes
    ----------
    X : np.ndarray
        Non-sensitive feature matrix after preprocessing.
    sensitive_features : np.ndarray
        Protected group labels (factorized) of each sensitive feature, shape
        (n_objects, n_sensitive_features).
    protected_group_names : list[list[str]]
        Original names of protected groups for each sensitive feature.
    dataset_balance : np.ndarray
        Balance of the dataset for each sensitive feature.
    batch_X : np.ndarray | None
        Reduced dataset composed of batch centers (representatives).
    batch_map : np.ndarray | None
        Map between batch objects and original objects.
    batch_weights : list[np.ndarray] | None
        Protected group counts per batch for each sensitive feature.
    batch_runtime : float
        Elapsed time for batch generation.

    Methods
    -------
    preprocess_data()
        Execute the full preprocessing pipeline.
    """

    dataset: str
    sensitive_names: list[str]
    binary: bool
    n_subsample: int | None
    n_batches: int | None
    n_features: int | None
    normalize: bool
    standardize: bool

    # UCI dataset IDs for automatic download
    UCI_DATASET_IDS = {
        "creditcard": 350,
        "bank_5k": 222,
        "bank_40k": 222,
        "adult": 2,
        "diabetes": 296,
        "census1990": 116,
    }

    def preprocess_data(self) -> None:
        """Execute the full preprocessing pipeline."""
        self._download_dataset()
        self._convert_features_to_numeric()
        self._subsample_features()
        self._extract_feature_names()
        self._factorize_sensitive_features()
        self._subsample_dataset()
        self._compute_dataset_balance()
        self._convert_to_arrays()
        self._scale_features()
        self._create_batches()
        self._save_preprocessed_dataset()

    def _download_dataset(self) -> None:
        """
        Download dataset from the UCI Machine Learning Repository or load it
        from the data/ folder.
        """
        if self.dataset not in self.UCI_DATASET_IDS:
            try:
                data = pd.read_csv(f"data/{self.dataset}.csv")
                self.X = data.drop(columns=self.sensitive_names)
                self.sensitive_features = data[self.sensitive_names]
            except FileNotFoundError as e:
                raise ValueError(
                    f"Dataset '{self.dataset}' was not found locally in 'data/' "
                    f"and is not available for download from the UCI repository "
                    f"(supported UCI datasets: {list(self.UCI_DATASET_IDS.keys())})."
                ) from e
        else:
            try:
                Path("data").mkdir(exist_ok=True)
                uci_id = self.UCI_DATASET_IDS[self.dataset]
                data = fetch_ucirepo(id=uci_id).data.features
                data.to_csv(f"data/{self.dataset}_raw.csv", index=False)
                self.X = data.drop(columns=self.sensitive_names)
                self.sensitive_features = data[self.sensitive_names]
            except Exception as e:
                raise RuntimeError(f"Failed to download dataset: {e}")

    def _convert_features_to_numeric(self) -> None:
        """Convert features to numeric, drop non-convertible columns."""
        self.X = self.X.apply(pd.to_numeric, errors="coerce")
        self.X = self.X.dropna(axis=1, how="all")

    def _subsample_features(self) -> None:
        """Keep only first n_features columns if specified."""
        if self.n_features:
            self.X = self.X.iloc[:, : self.n_features]

    def _extract_feature_names(self) -> None:
        """Store non-sensitive feature names."""
        self.features = self.X.columns.tolist()

    def _factorize_sensitive_features(self) -> None:
        """
        Optionally keep only the two largest groups of each sensitive feature, then
        factorize protected group names.
        """
        if self.binary:
            # Keep objects from the two largest protected groups of every sensitive
            # feature (group sizes are computed on the full dataset)
            mask = np.ones(self.X.shape[0], dtype=bool)
            for name in self.sensitive_names:
                group_counts = self.sensitive_features[name].value_counts()
                largest_groups = group_counts.index[:2]
                mask &= self.sensitive_features[name].isin(largest_groups).to_numpy()
            self.X = self.X[mask]
            self.sensitive_features = self.sensitive_features[mask]

        # Factorizing after filtering keeps factors aligned with group names
        factors = []
        self.protected_group_names = []
        for name in self.sensitive_names:
            factor, groups = pd.factorize(self.sensitive_features[name])
            factors.append(factor)
            self.protected_group_names.append(groups.tolist())
        self.sensitive_features = np.column_stack(factors)

        if not self.n_subsample:
            self._show_data_summary()

    def _subsample_dataset(self) -> None:
        """
        Subsample dataset using stratified sampling to maintain protected group proportions.
        """
        if not self.n_subsample:
            return
        rng = np.random.default_rng(42)
        # Strata are the combinations of protected groups across sensitive features
        _, strata = np.unique(
            self.sensitive_features, axis=0, return_inverse=True
        )
        strata_labels, counts = np.unique(strata, return_counts=True)
        group_indices = {
            group: np.flatnonzero(strata == group)
            for group in strata_labels
        }
        proportions = counts / counts.sum()
        objects_per_group = np.floor(self.n_subsample * proportions).astype(
            int
        )
        # Adjust for rounding gap
        deficit = self.n_subsample - objects_per_group.sum()
        if deficit > 0:
            # Add objects to largest groups
            for i in np.argsort(-proportions)[:deficit]:
                objects_per_group[i] += 1
        sampled_indices = np.concatenate(
            [
                rng.choice(group_indices[group], size=n_samples, replace=False)
                for group, n_samples in zip(
                    strata_labels, objects_per_group
                )
                if n_samples > 0
            ]
        )
        self.X = self.X.iloc[sampled_indices]
        self.sensitive_features = self.sensitive_features[sampled_indices]
        self._show_data_summary()

    def _show_data_summary(self) -> None:
        """Log summary table of protected groups for each sensitive feature."""
        for m, sensitive_name in enumerate(self.sensitive_names):
            counts = np.bincount(self.sensitive_features[:, m])
            logger.info(
                f"{sensitive_name:<20} {'Factor':<10} {'# Objects':<10}"
            )
            logger.info("-" * 46)
            for idx, name in enumerate(self.protected_group_names[m]):
                logger.info(f"{name:<20} {idx:<10} {counts[idx]:<10}")

    def _compute_dataset_balance(self) -> None:
        """Compute dataset balance for each sensitive feature."""
        self.dataset_balance = np.zeros(self.sensitive_features.shape[1])
        for m in range(self.sensitive_features.shape[1]):
            group_counts = np.bincount(self.sensitive_features[:, m])
            self.dataset_balance[m] = group_counts.min() / group_counts.max()

    def _convert_to_arrays(self) -> None:
        """Convert to numpy arrays and optimize memory."""
        if not isinstance(self.X, np.ndarray):
            self.X = self.X.to_numpy()
        self.X = self.X.astype(np.float32)
        self.sensitive_features = self.sensitive_features.astype(np.int8)

    def _scale_features(self) -> None:
        """Apply non-sensitive feature scaling."""
        if self.normalize:
            self.X = preprocessing.MinMaxScaler().fit_transform(self.X)
        elif self.standardize:
            self.X = preprocessing.StandardScaler().fit_transform(self.X)

    def _create_batches(self) -> None:
        """Construct batches for S-MPFC using FAISS k-means."""
        if self.n_batches is None:
            self.batch_X = None
            self.batch_map = None
            self.batch_weights = None
            self.batch_runtime = 0
            return

        start_time = time.perf_counter()
        self.batch_X, self.batch_map, self.batch_weights = create_batches(
            X=self.X,
            sensitive_features=self.sensitive_features,
            n_batches=self.n_batches,
            seed=42,
        )
        self.batch_runtime = time.perf_counter() - start_time

    def _save_preprocessed_dataset(self) -> None:
        """Save preprocessed dataset to CSV."""
        combined_data = np.column_stack([self.X, self.sensitive_features])
        columns = self.features + self.sensitive_names
        output_path = f"data/{self.dataset}_processed.csv"
        pd.DataFrame(combined_data, columns=columns).to_csv(
            output_path, index=False
        )
