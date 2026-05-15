# MIT License
# Copyright (c) 2026 Claudio Mantuano, University of Bern
# Paper: https://arxiv.org/abs/2605.13759

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from fair_clustering.base import FairClustering
from fair_clustering.blp import BLPBasedHeuristic
from fair_clustering.exact import ExactApproaches
from fair_clustering.flow import FlowBasedHeuristic
from fair_clustering.plotting import plot_clustering
from fair_clustering.preprocessing import DataPreprocessor


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


@dataclass(frozen=True)
class ExperimentConfig:
    # Data
    dataset: str
    sensitive_name: str
    binary: bool
    n_subsample: int | None
    n_batches: int | None
    n_features: int | None
    normalize: bool
    standardize: bool

    # Algorithms
    mpfc: bool
    msflowfc: bool
    smpfc: bool
    miqcp: bool
    setvars: bool

    # Experiments
    n_clusters: list[int]
    n_seeds: int
    target: str
    tolerances: list[float]
    global_time_limit: int
    plot: bool


class ExperimentRunner:
    """
    Class implementing the experimental pipeline for evaluating fair clustering
    algorithms across multiple parameter configurations.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._setup_algorithms()
        self._validate_config()
        self._preprocess_data()
        self._initialize_results()
        self._compute_tot_experiments()
        self.global_counter = 0

    def run_experiment(self) -> dict:
        """Run all experiments and export results."""
        with logging_redirect_tqdm():
            self._progress = tqdm(
                total=self.n_experiments,
                desc="Progress",
                unit="run",
                ncols=70,
            )
            try:
                for algorithm, selected in self.selected_algorithms.items():
                    if selected:
                        self._run_algorithm(algorithm)
            finally:
                self._progress.close()

        Path("results").mkdir(exist_ok=True)
        results_df = pd.DataFrame(self.results)
        results_df.drop(columns=["Labels", "Centers"]).to_csv(
            f"results/{self.config.dataset}_results.csv", index=False
        )

        with open(f"results/{self.config.dataset}_results.pkl", "wb") as f:
            pickle.dump(self.results, f)
        return self.results

    def _run_algorithm(self, algorithm: str) -> None:
        """Run experiments for all parameter combinations for an algorithm."""
        for k in self.config.n_clusters:
            for tolerance in self.config.tolerances:
                seeds = (
                    self.seeds
                    if algorithm in {"mpfc", "msflowfc", "smpfc"}
                    else [None]
                )
                time_limit = self._adjust_time_limit(algorithm)
                local_counter = 0

                for seed in seeds:
                    clustering, time_limit = self._run_instance(
                        algorithm, k, tolerance, seed, time_limit
                    )
                    local_counter += 1
                    if time_limit < 0:
                        self._handle_timeout(
                            clustering, algorithm, k, tolerance, seed
                        )
                        break
                    result_row = self._build_result_row(
                        clustering, algorithm, k, tolerance, seed
                    )
                    self._append_result_row(result_row)

                if algorithm in {"mpfc", "msflowfc", "smpfc"}:
                    self._append_result_summary(
                        algorithm, k, tolerance, local_counter
                    )

    def _run_instance(
        self,
        algorithm: str,
        k: int,
        tolerance: float,
        seed: int,
        time_limit: float,
    ) -> tuple[FairClustering, float]:
        """Run experiment on a single instance and return results."""
        self.global_counter += 1

        logger.info(
            "\n[%d/%d] dataset=%s algorithm=%s k=%d tolerance=%.2f seed=%s",
            self.global_counter,
            self.n_experiments,
            self.config.dataset,
            algorithm,
            k,
            tolerance,
            seed,
        )
        self._progress.update(1)

        clustering = self._create_instance(
            algorithm, k, tolerance, seed, time_limit
        )
        clustering_method = getattr(clustering, clustering.algorithm)
        clustering_method()
        return clustering, time_limit - clustering.runtime

    def _create_instance(
        self,
        algorithm: str,
        k: int,
        tolerance: float,
        seed: int,
        time_limit: float,
    ) -> FairClustering:
        """Create an instance given data, algorithm, and parameters."""
        args = {
            "X": self.X,
            "sensitive_feature": self.sensitive_feature,
            "tolerance": tolerance,
            "n_clusters": k,
            "target": self.config.target,
            "time_limit": time_limit,
            "seed": seed,
            "batch_X": self.batch_X,
            "batch_map": self.batch_map,
            "batch_weights": self.batch_weights,
            "algorithm": algorithm,
        }
        if algorithm in {"miqcp", "setvars"}:
            return ExactApproaches(**args)
        elif algorithm in {"mpfc", "smpfc"}:
            return BLPBasedHeuristic(**args)
        elif algorithm in {"msflowfc"}:
            return FlowBasedHeuristic(**args)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    def _adjust_time_limit(self, algorithm: str) -> int:
        """Compute time limit adjusted for preprocessing overhead."""
        if algorithm == "smpfc":
            return self.config.global_time_limit - self.batch_runtime
        return self.config.global_time_limit

    def _handle_timeout(
        self,
        clustering: FairClustering,
        algorithm: str,
        k: int,
        tolerance: float,
        seed: int,
    ) -> None:
        """Handle experiment timeout based on the algorithm."""
        logger.warning("The time limit has been exceeded. Terminating...")
        if algorithm in {"mpfc", "smpfc", "miqcp", "setvars"}:
            result_row = self._build_result_row(
                clustering, algorithm, k, tolerance, seed
            )
            self._append_result_row(result_row)
        elif algorithm in {"msflowfc"}:
            self._append_timeout_row(
                algorithm, k, tolerance, seed, clustering.runtime
            )

    def _build_result_row(
        self,
        clustering: FairClustering,
        algorithm: str,
        k: int,
        tolerance: float,
        seed: int,
    ) -> list[Any]:
        """Build a result row from clustering output."""
        return [
            self.config.dataset,
            self.X.shape[0],
            self.batch_X.shape[0] if algorithm == "smpfc" else None,
            self.features,
            self.config.sensitive_name,
            self.protected_group_names,
            self._get_preprocessing(),
            algorithm,
            seed,
            clustering.n_iter,
            k,
            tolerance,
            clustering.status,
            clustering.mipgap,
            clustering.runtime,
            clustering.clustering_cost,
            self.dataset_balance,
            clustering.target_balance,
            clustering.clustering_balance,
            clustering.cluster_balances,
            clustering.violation,
            clustering.excess,
            clustering.clustering_labels.tolist(),
            clustering.clustering_centers.tolist(),
        ]

    def _append_timeout_row(
        self,
        algorithm: str,
        k: int,
        tolerance: float,
        seed: int | None,
        runtime: float,
    ) -> None:
        """Append a timeout row to the dictionary of results."""
        row = [
            self.config.dataset,
            self.X.shape[0],
            self.batch_X.shape[0] if algorithm == "smpfc" else None,
            self.features,
            self.config.sensitive_name,
            self.protected_group_names,
            self._get_preprocessing(),
            algorithm,
            seed,
            None,
            k,
            tolerance,
            "timeout",
            None,
            runtime,
            None,
            self.dataset_balance,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ]
        self._append_result_row(row)

    def _append_result_summary(
        self, algorithm: str, k: int, tolerance: float, window_size: int
    ) -> None:
        """Append summary values for a given instance."""
        columns = list(self.results.keys())

        def compute_avg(column: str) -> float | None:
            """Compute average results across different random seeds."""
            values = [
                val
                for val in self.results[column][-window_size:]
                if val is not None
            ]
            return sum(values) / len(values) if values else None

        average_row = [
            self.config.dataset,
            self.X.shape[0],
            self.batch_X.shape[0] if algorithm == "smpfc" else None,
            self.features,
            self.config.sensitive_name,
            self.protected_group_names,
            self._get_preprocessing(),
            algorithm,
            "average",
            compute_avg("Iterations"),
            k,
            tolerance,
            None,
            compute_avg("MIPGap"),
            compute_avg("Running time (s)"),
            compute_avg("Clustering cost"),
            self.dataset_balance,
            compute_avg("Target balance"),
            compute_avg("Clustering balance"),
            None,
            compute_avg("Target balance violation (%)"),
            compute_avg("Target balance excess (%)"),
            None,
            None,
        ]
        self._append_result_row(average_row)

        best_row = self._find_best_result(window_size, algorithm, columns)
        for col in columns:
            self.results[col].append(best_row.get(col, None))
        if (
            best_row["Clustering cost"] is not None
            and self.X.shape[1] == 2
            and self.config.plot
        ):
            self._plot_clustering(algorithm, k, tolerance, best_row)

    def _get_preprocessing(self) -> str | None:
        """Get the name of the preprocessing method used."""
        if self.config.normalize:
            return "Normalization"
        elif self.config.standardize:
            return "Standardization"
        return None

    def _append_result_row(self, row: list[Any]) -> None:
        """Append a row to the dictionary of results."""
        for col, value in zip(self.results.keys(), row):
            self.results[col].append(value)

    def _find_best_result(
        self, window_size: int, algorithm: str, columns: list[str]
    ) -> dict[str, Any]:
        """Find the best (minimum cost) result in the current window."""
        costs = self.results["Clustering cost"][-(window_size + 1) : -1]
        valid_costs = [
            (i, val) for i, val in enumerate(costs) if val is not None
        ]

        idx = min(valid_costs, key=lambda x: x[1])[0] if valid_costs else 0
        absolute_idx = (
            idx + len(self.results["Clustering cost"]) - (window_size + 1)
        )
        row = {col: self.results[col][absolute_idx] for col in columns}
        row["Seed"] = str(row["Seed"]) + "_best"

        runtimes = self.results["Running time (s)"][-(window_size + 1) : -1]
        total_runtime = sum(t for t in runtimes if t is not None)
        row["Running time (s)"] = (
            total_runtime + self.batch_runtime
            if algorithm == "smpfc"
            else total_runtime
        )
        return row

    def _plot_clustering(
        self, algorithm: str, k: int, tolerance: float, row: dict[str, Any]
    ) -> None:
        """Generate and save a clustering visualization."""
        try:
            centers = np.array(row["Centers"])
            labels = np.array(row["Labels"])
            cluster_balances = np.array(row["Balance of clusters"])

            plot_clustering(
                self.config.dataset,
                self.X,
                self.config.sensitive_name,
                self.sensitive_feature,
                self.protected_group_names,
                algorithm,
                k,
                tolerance,
                labels,
                centers,
                row["Clustering cost"],
                row["Clustering balance"],
                cluster_balances,
            )

        except Exception as e:
            logger.warning("Could not plot clustering: %s", e)

    def _setup_algorithms(self) -> None:
        """Initialize algorithm selection and random seeds."""
        self.selected_algorithms = {
            "mpfc": self.config.mpfc,
            "msflowfc": self.config.msflowfc,
            "smpfc": self.config.smpfc,
            "miqcp": self.config.miqcp,
            "setvars": self.config.setvars,
        }
        self.seeds = list(range(1, self.config.n_seeds + 1))

    def _preprocess_data(self) -> None:
        """Load and preprocess dataset."""
        preprocessing = DataPreprocessor(
            dataset=self.config.dataset,
            sensitive_name=self.config.sensitive_name,
            binary=self.config.binary,
            n_subsample=self.config.n_subsample,
            n_batches=self.config.n_batches,
            n_features=self.config.n_features,
            normalize=self.config.normalize,
            standardize=self.config.standardize,
        )
        preprocessing.preprocess_data()

        self.X = preprocessing.X
        self.protected_group_names = preprocessing.protected_group_names
        self.sensitive_feature = preprocessing.sensitive_feature
        self.dataset_balance = preprocessing.dataset_balance
        self.features = preprocessing.features
        self.batch_X = preprocessing.batch_X
        self.batch_map = preprocessing.batch_map
        self.batch_weights = preprocessing.batch_weights
        self.batch_runtime = preprocessing.batch_runtime

    def _initialize_results(self) -> None:
        """Initialize the dictionary of results."""
        columns = [
            "Dataset",
            "Objects",
            "Batches",
            "Features",
            "Sensitive feature",
            "Protected groups",
            "Preprocessing",
            "Algorithm",
            "Seed",
            "Iterations",
            "k",
            "Tolerance",
            "Status",
            "MIPGap",
            "Running time (s)",
            "Clustering cost",
            "Dataset balance",
            "Target balance",
            "Clustering balance",
            "Balance of clusters",
            "Target balance violation (%)",
            "Target balance excess (%)",
            "Labels",
            "Centers",
        ]
        self.results = {col: [] for col in columns}

    def _validate_config(self) -> None:
        """Validate experiment configuration."""
        n_algorithms = sum(self.selected_algorithms.values())
        n_heuristics = sum(
            self.selected_algorithms[alg]
            for alg in ("mpfc", "msflowfc", "smpfc")
        )
        if n_algorithms == 0:
            raise ValueError("Please select at least one algorithm.")
        if not self.config.n_clusters:
            raise ValueError("Please indicate the number of clusters.")
        if n_heuristics > 0:
            if not self.seeds:
                raise ValueError("Please indicate at least one random seed.")
            if not self.config.tolerances:
                raise ValueError(
                    "Please indicate at least one tolerance value."
                )
        if self.config.smpfc and self.config.n_batches is None:
            raise ValueError(
                "The algorithm S-MPFC is selected, but n_batches is None."
            )
        if self.config.smpfc and any(
            self.config.n_batches < k for k in self.config.n_clusters
        ):
            raise ValueError(
                "n_batches must be greater than or equal to n_clusters."
            )
        if self.config.normalize and self.config.standardize:
            raise ValueError(
                "Both normalization and standardization are selected. "
                "Please select only one of them."
            )

    def _compute_tot_experiments(self) -> None:
        """Compute the total number of experiments for progress tracking."""
        n_heuristic = sum(
            self.selected_algorithms[alg]
            for alg in ("mpfc", "msflowfc", "smpfc")
        ) * len(self.seeds)
        n_exact = (
            self.selected_algorithms["miqcp"]
            + self.selected_algorithms["setvars"]
        )
        self.n_experiments = (
            (n_heuristic + n_exact)
            * len(self.config.tolerances)
            * len(self.config.n_clusters)
        )
