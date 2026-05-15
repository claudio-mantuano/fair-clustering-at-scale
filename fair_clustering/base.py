# MIT License
# Copyright (c) 2026 Claudio Mantuano, University of Bern
# Paper: https://arxiv.org/abs/2605.13759

import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import kmeans_plusplus


@dataclass
class FairClustering:
    """
    Base class for fair clustering algorithms.

    Parameters
    ----------
    X : np.ndarray
        Non-sensitive feature matrix, shape (n_objects, n_features).
    sensitive_feature : np.ndarray
        Protected group labels, shape (n_objects,).
    tolerance : float
        Fairness tolerance parameter in [0, 1], or [0,1) for MS-FlowFC.
    n_clusters : int
        Number of clusters to be identified.

    target : str, default="dataset"
        Use 'maximum' to compute the target balance relative to the maximum balance,
        or 'dataset' to compute it relative to the dataset balance.
    time_limit : float, default=3600
        Maximum runtime in seconds per instance.
    seed : int, default=42
        Random seed for reproducibility.
    batch_X : np.ndarray | None, default=None
        Non-sensitive feature matrix for S-MPFC, shape (n_batches, n_features).
    batch_map : np.ndarray | None, default=None
        Mapping from batches (representatives) to indices of the original dataset.
    batch_weights : np.ndarray | None, default=None
        Number of objects from each protected group in the batches.

    Attributes
    ----------
    protected_groups : list[np.ndarray]
        List of arrays containing indices of objects in each group sorted by size.
    dataset_balance : float
        Balance of the dataset.
    max_balance : float
        Maximum balance achievable given data and number of clusters.
    target_balance : float
        Target balance for the fair clustering algorithms.
    clustering_labels : np.ndarray
        Cluster assignments with values in [0, n_clusters-1], shape (n_objects,).
    clustering_centers : np.ndarray
        Coordinate array of cluster centers, shape (n_clusters, n_features).

    n_iter : int
        Number of iterations performed by the algorithm.
    runtime : float
        Running time of the algorithm in seconds.
    algorithm : str
        Name of the algorithm (only "mpfc" or "smpfc").
    clustering_cost : float
        Total within-cluster squared Euclidean distance.
    clustering_balance : float | None
        Minimum balance across all clusters.
    cluster_balances : list[float]
        List of balance values for each cluster.
    cluster_sizes : list[int]
        Number of objects in each cluster.
    violation : float | None
        Percentage violation of target balance (fairness constraint violation).
    excess : float | None
        Percentage excess with respect to target balance.
    status : int | None
        Status code returned by the solver.
    mipgap : float | None
        MIP Gap returned by the solver.
    """

    # Parameters (required)
    X: np.ndarray
    sensitive_feature: np.ndarray
    tolerance: float
    n_clusters: int

    # Parameters (optional)
    target: str = "dataset"
    time_limit: float = 3600
    seed: int = 42
    batch_X: np.ndarray | None = None
    batch_map: np.ndarray | None = None
    batch_weights: np.ndarray | None = None

    # Attributes (computed/initialized in __post_init__)
    protected_groups: list[np.ndarray] = field(init=False)
    dataset_balance: float = field(init=False)
    max_balance: float = field(init=False)
    target_balance: float = field(init=False)
    clustering_labels: np.ndarray = field(init=False)
    clustering_centers: np.ndarray = field(init=False)

    # Attributes (populated by the algorithms)
    n_iter: int = 0
    runtime: float = 0.0
    algorithm: str | None = None
    clustering_cost: float = float("inf")
    clustering_balance: float | None = None
    cluster_balances: list[float] = field(default_factory=list)
    cluster_sizes: list[int] = field(default_factory=list)
    violation: float | None = None
    excess: float | None = None
    status: int | None = None
    mipgap: float | None = None

    def __post_init__(self):
        """Initialize class attributes and compute target balance."""
        self._initialize_attributes()
        self._compute_dataset_balance()
        self._compute_max_balance()
        self._compute_target_balance()

    def _initialize_attributes(self) -> None:
        """Initialize clustering labels, centers, and protected group indices."""
        self.clustering_labels = np.full(
            (self.X.shape[0],), -1, dtype=int
        )
        self.clustering_centers = np.full(
            (self.n_clusters, self.X.shape[1]), -1.0, dtype=float
        )
        group_labels = np.unique(self.sensitive_feature)
        group_size = {
            g: np.sum(self.sensitive_feature == g) for g in group_labels
        }
        # Labels of protected groups are sorted by size (largest first)
        sorted_groups = sorted(
            group_labels, key=lambda g: group_size[g], reverse=True
        )
        self.protected_groups = [
            np.flatnonzero(self.sensitive_feature == g) for g in sorted_groups
        ]

    def _compute_dataset_balance(self) -> None:
        """Compute dataset balance for the specified sensitive feature."""
        group_counts = np.bincount(self.sensitive_feature)
        self.dataset_balance = group_counts.min() / group_counts.max()

    def _compute_max_balance(self) -> None:
        """Compute the maximum balance achievable given data and number of clusters."""
        n_smallest_group = self.protected_groups[-1].shape[0]
        n_largest_group = self.protected_groups[0].shape[0]
        self.max_balance = np.floor(
            n_smallest_group / self.n_clusters
        ) / np.ceil(n_largest_group / self.n_clusters)

    def _compute_target_balance(self) -> None:
        """Compute the target balance based on the tolerance parameter."""
        if self.target == "maximum":
            self.target_balance = (1 - self.tolerance) * self.max_balance
        elif self.target == "dataset":
            self.target_balance = (1 - self.tolerance) * self.dataset_balance
        else:
            raise ValueError(
                f"Invalid target '{self.target}'. Use 'maximum' or 'dataset'."
            )

    def _extract_results(self) -> None:
        """Compute and extract all result metrics."""
        self.clustering_centers, _ = self._update_centers(
            self.X, self.clustering_labels, self.n_clusters
        )
        self.clustering_cost = self._get_cost(
            self.X, self.clustering_centers, self.clustering_labels
        )
        self.clustering_balance, self.cluster_balances = (
            self._get_clustering_balance(
                self.sensitive_feature, self.clustering_labels
            )
        )
        self.cluster_sizes = self._get_cluster_sizes(self.clustering_labels)
        # Percentage gap between clustering balance and target balance
        bal_gap = (
            (self.target_balance - self.clustering_balance)
            / self.target_balance
        ) * 100
        # How much below target balance (0 if higher or equal)
        self.violation = max(bal_gap, 0)
        # How much above target balance (0 if below)
        self.excess = max(-bal_gap, 0)

    @staticmethod
    def _initialize_centers_kmeans_pp(
        X: np.ndarray, n_centers: int, seed: int
    ) -> np.ndarray:
        """
        Initialize cluster centers using the k-means++ algorithm.

        Parameters
        ----------
        X : np.ndarray
            Non-sensitive feature matrix, shape (n_objects, n_features).
        n_centers : int
            Number of cluster centers to initialize.
        seed : int
            Random seed.

        Returns
        -------
        initial_centers : np.ndarray
            Coordinate array of initial centers, shape (n_centers, n_features).
        """
        initial_centers, _ = kmeans_plusplus(
            X=X, n_clusters=n_centers, random_state=seed
        )
        return initial_centers

    @staticmethod
    def _compute_distance_matrix(
        X: np.ndarray, centers: np.ndarray
    ) -> np.ndarray:
        """
        Compute matrix of distances using vectorized operations and the algebraic identity
        for squared Euclidean distance. This vectorized approach is faster than computing
        distances element-wise for large datasets.

        Parameters
        ----------
        X : np.ndarray
            Non-sensitive feature matrix, shape (n_objects, n_features).
        centers : np.ndarray
            Coordinate array of cluster centers, shape (n_centers, n_features).

        Returns
        -------
        distance_matrix : np.ndarray
            Array where element [i, j] contains the squared Euclidean distance between 
            object i and cluster center j, shape (n_objects, n_centers).
        """
        squared_norm_objects = np.sum(X**2, axis=1).reshape(-1, 1)
        squared_norm_centers = np.sum(centers**2, axis=1).reshape(1, -1)
        dot_product = np.dot(X, centers.T)
        distance_matrix = (
            squared_norm_objects + squared_norm_centers - 2 * dot_product
        )
        return distance_matrix

    @staticmethod
    def _update_centers(
        X: np.ndarray, labels: np.ndarray, n_clusters: int
    ) -> tuple[np.ndarray, float]:
        """
        Update cluster centers by computing the mean of non-sensitive features of
        assigned objects.

        Parameters
        ----------
        X : np.ndarray
            Non-sensitive feature matrix, shape (n_objects, n_features).
        labels : np.ndarray
            Cluster assignments with values in [0, n_clusters-1], shape (n_objects,).
        n_clusters : int
            Number of clusters.

        Returns
        -------
        updated_centers : np.ndarray
            Coordinate array of new cluster centers, shape (n_clusters, n_features).
        elapsed_time : float
            Elapsed time in seconds.
        """
        start_time = time.perf_counter()
        n_features = X.shape[1]

        coordinates_sum = np.zeros((n_clusters, n_features), dtype=X.dtype)
        n_objects_clusters = np.bincount(labels, minlength=n_clusters).reshape(
            -1, 1
        )
        np.add.at(coordinates_sum, labels, X)
        updated_centers = coordinates_sum / n_objects_clusters

        elapsed_time = time.perf_counter() - start_time
        return updated_centers, elapsed_time

    def _update_centers_weighted(
        self, X: np.ndarray, labels: np.ndarray, n_clusters: int
    ) -> tuple[np.ndarray, float]:
        """
        Update cluster centers by computing the weighted mean of non-sensitive features
        of assigned representatives.

        Parameters
        ----------
        X : np.ndarray
            Non-sensitive feature matrix, shape (n_representatives, n_features).
        labels : np.ndarray
            Cluster assignments with values in [0, n_clusters-1], shape (n_representatives,).
        n_clusters : int
            Number of clusters.

        Returns
        -------
        updated_centers : np.ndarray
            Coordinate array of new cluster centers, shape (n_clusters, n_features).
        elapsed_time : float
            Elapsed time in seconds.
        """
        start_time = time.perf_counter()
        n_features = X.shape[1]

        weights = self.batch_weights.sum(axis=1)
        coordinates_weighted_sum = np.zeros(
            (n_clusters, n_features), dtype=X.dtype
        )
        np.add.at(coordinates_weighted_sum, labels, X * weights[:, None])
        n_objects_clusters = np.bincount(
            labels, weights=weights, minlength=n_clusters
        ).reshape(-1, 1)

        updated_centers = coordinates_weighted_sum / n_objects_clusters
        elapsed_time = time.perf_counter() - start_time
        return updated_centers, elapsed_time

    @staticmethod
    def _get_cost(
        X: np.ndarray, centers: np.ndarray, labels: np.ndarray
    ) -> float:
        """
        Compute the total within-cluster squared Euclidean distance.

        Parameters
        ----------
        X : np.ndarray
            Non-sensitive feature matrix, shape (n_objects, n_features).
        centers : np.ndarray
            Coordinate array of cluster centers, shape (n_clusters, n_features).
        labels : np.ndarray
            Cluster assignments with values in [0, n_clusters-1], shape (n_objects,).

        Returns
        -------
        cost : float
            Total within-cluster sum of squared Euclidean distances.
        """
        distances = X - centers[labels]
        tot_distance = (distances**2).sum()
        return tot_distance

    @staticmethod
    def _get_clustering_balance(
        sensitive_feature: np.ndarray, labels: np.ndarray
    ) -> tuple[float, list[float]]:
        """
        Compute minimum balance across all clusters.

        Parameters
        ----------
        sensitive_feature : np.ndarray
            Protected group labels, shape (n_objects,).
        labels : np.ndarray
            Cluster assignment for each object, shape (n_objects,).

        Returns
        -------
        clustering_balance : float
            Minimum balance across all clusters.
        balances : list[float]
            List of balance values for each cluster.
        """
        n_clusters = labels.max() + 1
        n_groups = sensitive_feature.max() + 1

        # Number of objects from each group in each cluster
        counts, _, _ = np.histogram2d(
            labels,
            sensitive_feature,
            bins=[n_clusters, n_groups],
            range=[(0, n_clusters), (0, n_groups)],
        )

        max_counts = counts.max(axis=1)
        max_counts[max_counts == 0] = 1  # no division by 0
        min_counts = counts.min(axis=1)

        balances = min_counts / max_counts
        clustering_balance = balances.min()
        cluster_balances = balances.tolist()
        return clustering_balance, cluster_balances
    
    @staticmethod
    def _get_cluster_sizes(
        labels: np.ndarray
    ) -> list[int]:
        """
        Compute the size of all clusters.

        Parameters
        ----------
        labels : np.ndarray
            Cluster assignment for each object, shape (n_objects,).

        Returns
        -------
        cluster_sizes : list[int]
            Number of objects in each cluster.
        """
        _, cluster_sizes = np.unique(labels, return_counts=True)
        return cluster_sizes.tolist()
