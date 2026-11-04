# MIT License
# Copyright (c) 2026 Claudio Mantuano, University of Bern
# Paper: https://arxiv.org/abs/2605.13759

import time

import numpy as np
from sklearn.cluster import kmeans_plusplus

from fair_clustering.milp import MILPBasedHeuristic
from fair_clustering.flow import FlowBasedHeuristic
from fair_clustering.exact import ExactApproaches


class FairClustering(MILPBasedHeuristic, FlowBasedHeuristic, ExactApproaches):
    """
    Base class for fair k-means clustering.

    Parameters
    ----------
    algorithm : str
        Algorithm to run ("mpfc", "smpfc", "msflowfc", "miqcp", or "setvars").
    n_clusters : int
        Number of clusters to be identified.
    tolerance : float
        Fairness tolerance parameter in [0, 1], or [0, 1) for MS-FlowFC.
    target : str, default="dataset"
        Use 'maximum' to compute the target balance relative to the maximum 
        balance, or 'dataset' to compute it relative to the dataset balance.
    time_limit : float, default=3600
        Maximum runtime in seconds per instance.
    seed : int, default=42
        Random seed for reproducibility.
    solver : str, default="scip"
        MIP solver used by MPFC and S-MPFC: "scip" (open source, no commercial 
        license required) or "gurobi" (requires a commercial license). Ignored 
        by the other algorithms.

    Attributes
    ----------
    labels_ : np.ndarray
        Cluster assignments with values in [0, n_clusters-1], shape (n_objects,).
    cluster_centers_ : np.ndarray
        Coordinate array of cluster centers, shape (n_clusters, n_features).
    cost_ : float
        Total within-cluster squared Euclidean distance.
    balance_ : np.ndarray | None
        Minimum balance across all clusters for each sensitive feature, shape
        (n_sensitive_features,).
    cluster_balances_ : np.ndarray
        Balance of each cluster for each sensitive feature, shape
        (n_sensitive_features, n_clusters).
    cluster_sizes_ : list[int]
        Number of objects in each cluster.
    n_iter_ : int
        Number of iterations performed by the algorithm.
    total_runtime_ : float
        Running time of the algorithm in seconds.
    external_runtime_ : float | None
        Running time spent in the calls of external compiled libraries in seconds. 
        None for the exact approaches.
    python_runtime_ : float | None
        Running time spent in Python code in seconds. None for the exact approaches.
    violation_ : np.ndarray | None
        Percentage violation of target balance (fairness constraint violation) for
        each sensitive feature, shape (n_sensitive_features,).
    excess_ : np.ndarray | None
        Percentage excess with respect to target balance for each sensitive feature,
        shape (n_sensitive_features,).
    status_ : int | str | None
        Status returned by the solver (Gurobi code, or SCIP or Hexaly status string).
    bound_ : float | None
        Best lower bound (LB) on the cost returned by the solver.
    gap_ : float | None
        Relative gap (UB - LB) / UB between the cost and the best bound.
    protected_groups_ : list[list[np.ndarray]]
        For each sensitive feature, list of arrays containing indices of objects in
        each group sorted by size.
    dataset_balance_ : np.ndarray
        Balance of the dataset for each sensitive feature, shape (n_sensitive_features,).
    max_balance_ : np.ndarray
        Maximum balance achievable given data and number of clusters for each
        sensitive feature, shape (n_sensitive_features,).
    target_balance_ : np.ndarray
        Target balance for the fair clustering algorithms for each sensitive feature,
        shape (n_sensitive_features,).
    """

    _SUPPORTED_ALGORITHMS = ("mpfc", "smpfc", "msflowfc", "miqcp", "setvars")
    _SUPPORTED_SOLVERS = ("scip", "gurobi")

    def __init__(
        self,
        algorithm: str,
        n_clusters: int,
        tolerance: float,
        target: str = "dataset",
        time_limit: float = 3600,
        seed: int = 42,
        solver: str = "scip",
    ):
        if algorithm not in self._SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm '{algorithm}'. "
                f"Choose from {self._SUPPORTED_ALGORITHMS}."
            )
        if solver not in self._SUPPORTED_SOLVERS:
            raise ValueError(
                f"Unsupported solver '{solver}'. "
                f"Choose from {self._SUPPORTED_SOLVERS}."
            )
        self.algorithm = algorithm
        self.n_clusters = n_clusters
        self.tolerance = tolerance
        self.target = target
        self.time_limit = time_limit
        self.seed = seed
        self.solver = solver

    def fit(
        self,
        X: np.ndarray,
        sensitive_features: np.ndarray,
        batch_X: np.ndarray | None = None,
        batch_map: np.ndarray | None = None,
        batch_weights: list[np.ndarray] | None = None,
    ) -> "FairClustering":
        """
        Run the selected fair clustering algorithm on the given data.

        Parameters
        ----------
        X : np.ndarray
            Non-sensitive feature matrix, shape (n_objects, n_features).
        sensitive_features : np.ndarray
            Protected group labels of each sensitive feature, shape
            (n_objects, n_sensitive_features), or shape (n_objects,) for a single
            sensitive feature. MS-FlowFC supports only a single sensitive feature.
        batch_X : np.ndarray | None, default=None
            Non-sensitive feature matrix for S-MPFC, shape (n_batches, n_features).
        batch_map : np.ndarray | None, default=None
            Mapping from batches (representatives) to indices of the original dataset.
        batch_weights : list[np.ndarray] | None, default=None
            Number of objects from each protected group in the batches, one array
            of shape (n_batches, n_groups) per sensitive feature.

        Returns
        -------
        self : FairClustering
            Fitted instance with populated result attributes.
        """
        self.X = X
        # A single sensitive feature can also be passed as a 1D array
        if sensitive_features.ndim == 1:
            sensitive_features = sensitive_features.reshape(-1, 1)
        self.sensitive_features = sensitive_features
        self.batch_X = batch_X
        self.batch_map = batch_map
        self.batch_weights = batch_weights

        self._reset_results()
        self._initialize_attributes()
        self._compute_dataset_balance()
        self._compute_max_balance()
        self._compute_target_balance()

        getattr(self, f"_{self.algorithm}")()
        if self.external_runtime_ is not None:
            self.python_runtime_ = self.total_runtime_ - self.external_runtime_
        return self

    def _reset_results(self) -> None:
        """Reset the result attributes populated by the algorithms."""
        self.n_iter_ = 0
        self.total_runtime_ = 0.0
        self.external_runtime_ = None
        self.python_runtime_ = None
        self.cost_ = float("inf")
        self.balance_ = None
        self.cluster_balances_ = []
        self.cluster_sizes_ = []
        self.violation_ = None
        self.excess_ = None
        self.status_ = None
        self.bound_ = None
        self.gap_ = None

    def _initialize_attributes(self) -> None:
        """Initialize clustering labels, centers, and protected group indices."""
        self.labels_ = np.full(
            (self.X.shape[0],), -1, dtype=int
        )
        self.cluster_centers_ = np.full(
            (self.n_clusters, self.X.shape[1]), -1.0, dtype=float
        )
        self.protected_groups_ = []
        for m in range(self.sensitive_features.shape[1]):
            sensitive_feature = self.sensitive_features[:, m]
            group_labels = np.unique(sensitive_feature)
            group_size = {
                g: np.sum(sensitive_feature == g) for g in group_labels
            }
            # Labels of protected groups are sorted by size (largest first)
            sorted_groups = sorted(
                group_labels, key=lambda g: group_size[g], reverse=True
            )
            self.protected_groups_.append(
                [np.flatnonzero(sensitive_feature == g) for g in sorted_groups]
            )

    def _compute_dataset_balance(self) -> None:
        """Compute dataset balance for each sensitive feature."""
        self.dataset_balance_ = np.zeros(self.sensitive_features.shape[1])
        for m in range(self.sensitive_features.shape[1]):
            group_counts = np.bincount(self.sensitive_features[:, m])
            self.dataset_balance_[m] = group_counts.min() / group_counts.max()

    def _compute_max_balance(self) -> None:
        """
        Compute the maximum balance achievable given data and number of clusters
        for each sensitive feature.
        """
        self.max_balance_ = np.zeros(len(self.protected_groups_))
        for m, protected_groups in enumerate(self.protected_groups_):
            n_smallest_group = protected_groups[-1].shape[0]
            n_largest_group = protected_groups[0].shape[0]
            self.max_balance_[m] = np.floor(
                n_smallest_group / self.n_clusters
            ) / np.ceil(n_largest_group / self.n_clusters)

    def _compute_target_balance(self) -> None:
        """
        Compute the target balance for each sensitive feature based on the
        tolerance parameter.
        """
        if self.target == "maximum":
            self.target_balance_ = (1 - self.tolerance) * self.max_balance_
        elif self.target == "dataset":
            self.target_balance_ = (1 - self.tolerance) * self.dataset_balance_
        else:
            raise ValueError(
                f"Invalid target '{self.target}'. Use 'maximum' or 'dataset'."
            )

    def _extract_results(self) -> None:
        """Compute and extract all result metrics."""
        self.cluster_centers_, _ = self._update_centers(
            self.X, self.labels_, self.n_clusters
        )
        self.cost_ = self._get_cost(
            self.X, self.cluster_centers_, self.labels_
        )
        self.balance_, self.cluster_balances_ = (
            self._get_clustering_balance(
                self.sensitive_features, self.labels_
            )
        )
        self.cluster_sizes_ = self._get_cluster_sizes(self.labels_)
        # Percentage gap between clustering balance and target balance
        bal_gap = (
            (self.target_balance_ - self.balance_)
            / self.target_balance_
        ) * 100
        # How much below target balance (0 if higher or equal)
        self.violation_ = np.maximum(bal_gap, 0)
        # How much above target balance (0 if below)
        self.excess_ = np.maximum(-bal_gap, 0)
        # Gap between cost (UB) and best bound (LB), only for exact approaches
        if self.bound_ is not None:
            self.gap_ = ((self.cost_ - self.bound_) / self.cost_) * 100

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

        # Batch sizes (identical for every sensitive feature)
        weights = self.batch_weights[0].sum(axis=1)
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
        sensitive_features: np.ndarray, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute minimum balance across all clusters for each sensitive feature.

        Parameters
        ----------
        sensitive_features : np.ndarray
            Protected group labels of each sensitive feature, shape
            (n_objects, n_sensitive_features).
        labels : np.ndarray
            Cluster assignment for each object, shape (n_objects,).

        Returns
        -------
        clustering_balance : np.ndarray
            Minimum balance across all clusters for each sensitive feature, shape
            (n_sensitive_features,).
        cluster_balances : np.ndarray
            Balance of each cluster for each sensitive feature, shape
            (n_sensitive_features, n_clusters).
        """
        n_clusters = labels.max() + 1
        cluster_balances = np.zeros((sensitive_features.shape[1], n_clusters))

        for m in range(sensitive_features.shape[1]):
            n_groups = sensitive_features[:, m].max() + 1

            # Number of objects from each group in each cluster
            counts, _, _ = np.histogram2d(
                labels,
                sensitive_features[:, m],
                bins=[n_clusters, n_groups],
                range=[(0, n_clusters), (0, n_groups)],
            )

            max_counts = counts.max(axis=1)
            max_counts[max_counts == 0] = 1  # no division by 0
            min_counts = counts.min(axis=1)

            cluster_balances[m] = min_counts / max_counts

        clustering_balance = cluster_balances.min(axis=1)
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
