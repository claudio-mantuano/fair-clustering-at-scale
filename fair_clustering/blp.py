# MIT License
# Copyright (c) 2026 Claudio Mantuano, University of Bern
# Paper: https://arxiv.org/abs/2605.13759

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np
import pyscipopt as scip

if TYPE_CHECKING:
    import gurobipy as gb


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


class BLPBasedHeuristic:
    """
    Class implementing the heuristics relying on a MIP solver (SCIP or Gurobi). The 
    algorithms alternate between binary linear programming-based assignment (BLP 
    with fixed centers) and cluster center update (with fixed assignments), according 
    to the k-means decomposition scheme.

    Provides the algorithms "mpfc" and "smpfc" to the FairClustering class defined 
    in fair_clustering.base, which invokes them through `fit()`.

    Methods
    -------
    _mpfc()
        Solve the fair k-means clustering problem using the Mathematical
        Programming-based Fair Clustering algorithm (MPFC).
    _smpfc()
        Solve the fair k-means clustering problem using the Scalable Mathematical
        Programming-based Fair Clustering algorithm (S-MPFC) with batches.
    """

    def _mpfc(self) -> None:
        """Run the MPFC algorithm."""
        self.labels_, self.status_ = self._run_decomposition()
        if self.status_ is None:
            self._extract_results()

    def _smpfc(self) -> None:
        """Run the S-MPFC algorithm."""
        self._validate_batches()
        batch_labels, self.status_ = self._run_decomposition()
        if self.status_ is None:
            self._backmap(batch_labels=batch_labels)
            self._extract_results()

    def _validate_batches(self) -> None:
        """Ensure batch-related parameters are provided for S-MPFC."""
        if (
            self.batch_X is None
            or self.batch_map is None
            or self.batch_weights is None
        ):
            raise ValueError(
                "S-MPFC requires `batch_X`, `batch_map`, and `batch_weights`. "
                "Use `create_batches` from `fair_clustering.preprocessing` to construct them."
            )

    def _backmap(self, batch_labels: np.ndarray) -> None:
        """Map batch assignments to original objects."""
        self.labels_ = batch_labels[self.batch_map]

    def _run_decomposition(
        self, max_iter: int = 100, min_improvement: float = 0.1
    ) -> tuple[np.ndarray | None, int | str | None]:
        """
        Implement the k-means decomposition scheme (initialization, assignment, and
        center update) to cluster objects until a stopping criterion is met
        (i.e., minimum cost improvement is not achieved, time limit is exceeded,
        or maximum number of iterations is attained).

        Parameters
        ----------
        max_iter : int, default=100
            Maximum number of iterations (one iteration is a pair assignment-update).
        min_improvement : float, default=0.1
            Minimum cost improvement (%) required to continue.

        Returns
        -------
        best_labels : np.ndarray
            Best assignments, or -1 array if no solution.
        status : int | str | None
            Solver status if failed, None otherwise.
        """
        X = self.X if self.algorithm == "mpfc" else self.batch_X
        best_labels = np.full((X.shape[0],), -1, dtype=np.int32)
        best_cost = float("inf")
        start_time = time.perf_counter()

        centers = self._initialize_centers_kmeans_pp(
            X=X, n_centers=self.n_clusters, seed=self.seed
        )

        while self.n_iter_ < max_iter:
            # Assignment step
            labels, assignment_runtime, status = self._assign_objects(
                X=X,
                centers=centers,
                initial_labels=best_labels,
                time_limit=self.time_limit,
                iter=self.n_iter_,
            )
            if status is not None:
                self.cost_ = None
                self.runtime_ = time.perf_counter() - start_time
                return best_labels, status

            # Update step
            if self.algorithm == "mpfc":
                centers, update_runtime = self._update_centers(
                    X=X, labels=labels, n_clusters=self.n_clusters
                )
            elif self.algorithm == "smpfc":
                centers, update_runtime = self._update_centers_weighted(
                    X=X, labels=labels, n_clusters=self.n_clusters
                )
            cost = self._get_cost(X=X, centers=centers, labels=labels)

            # Check stopping criteria
            if np.isinf(best_cost):
                cost_improvement = float("inf")
            else:
                cost_improvement = ((best_cost - cost) / best_cost) * 100
            elapsed_time = time.perf_counter() - start_time
            if (cost_improvement < min_improvement) or (
                elapsed_time > self.time_limit
            ):
                if np.any(best_labels == -1):
                    self.cost_ = None
                    self.runtime_ = time.perf_counter() - start_time
                    return best_labels, 9
                else:
                    break
            else:
                best_cost = cost
                best_labels = labels.copy()

            logger.info(
                "iter=%-2d update[s]=%6.6f  assignment[s]=%6.4f  %s=%6.4f",
                self.n_iter_,
                update_runtime,
                assignment_runtime,
                "cost" if self.algorithm == "mpfc" else "cost(reduced)",
                cost,
            )
            self.n_iter_ += 1

        self.runtime_ = time.perf_counter() - start_time
        return best_labels, None

    def _assign_objects(
        self,
        X: np.ndarray,
        centers: np.ndarray,
        initial_labels: np.ndarray,
        time_limit: float,
        iter: int,
    ) -> tuple[np.ndarray | None, float, int | str | None]:
        """
        Solve BLP with fixed centers to assign objects to clusters.

        Parameters
        ----------
        X : np.ndarray
            Non-sensitive feature matrix, shape (n_objects, n_features) for MPFC or 
            (n_representatives, n_features) for S-MPFC.
        centers : np.ndarray
            Coordinate array of cluster centers, shape (n_clusters, n_features).
        initial_labels : np.ndarray
            Cluster assignments from previous iteration (used for warm start).
        time_limit : float
            Solver time limit.

        Returns
        -------
        labels : np.ndarray | None
            Cluster assignments of shape (n_objects,) if solved, None otherwise.
        runtime : float
            Solver running time.
        status : int | str | None
            Solver status if failed (Gurobi code or SCIP status string),
            None if solved.
        """
        objects = (
            range(self.X.shape[0])
            if self.algorithm == "mpfc"
            else range(self.batch_X.shape[0])
        )
        clusters = range(self.n_clusters)
        distances = self._compute_distance_matrix(X=X, centers=centers)
        distances_dict = {
            (i, j): distances[i, j] for i in objects for j in clusters
        }

        setup_binary_linear_program = (
            self._setup_blp_gurobi
            if self.solver == "gurobi"
            else self._setup_blp_scip
        )
        model, x = setup_binary_linear_program(
            distances=distances_dict,
            initial_labels=initial_labels,
            solver_time_limit=time_limit,
            iter=iter,
        )
        start_time = time.perf_counter()
        model.optimize()
        runtime = time.perf_counter() - start_time

        if self.solver == "gurobi":
            if model.SolCount > 0:
                labels = np.fromiter(
                    (j for (i, j), var in x.items() if var.X > 0.5),
                    dtype=np.int32,
                    count=len(objects),
                )
                return labels, runtime, None
            else:
                return None, runtime, model.Status
        else:
            if model.getNSols() > 0:
                labels = np.fromiter(
                    (j for (i, j), var in x.items() if model.getVal(var) > 0.5),
                    dtype=np.int32,
                    count=len(objects),
                )
                return labels, runtime, None
            else:
                return None, runtime, model.getStatus()

    def _setup_blp_gurobi(
        self,
        distances: dict,
        initial_labels: np.ndarray,
        solver_time_limit: float,
        iter: int,
        warm_start: bool = False,
    ) -> tuple[gb.Model, gb.tupledict]:
        """
        Build the BLP model for the assignment of objects to cluster centers
        using Gurobi.

        Parameters
        ----------
        distances : dict
            {(i,j): distance from object i to cluster center j}.
        initial_labels : np.ndarray
            Cluster assignments from previous iteration (used for warm start).
        solver_time_limit : float
            Gurobi solver time limit.
        warm_start : bool, default=False
            Boolean parameter to trigger warm start.

        Returns
        -------
        model : gb.Model
            Built Gurobi model.
        x : gb.tupledict
            Cluster assignment variables.
        """
        try:
            import gurobipy as gb
        except ImportError as e:
            raise ImportError(
                "solver='gurobi' requires gurobipy, which is not installed. "
                "Install the gurobipy version matching your local Gurobi "
                "installation, or use solver='scip'."
            ) from e
        X = self.X if self.algorithm == "mpfc" else self.batch_X
        objects = range(X.shape[0])
        clusters = range(self.n_clusters)

        model = gb.Model()
        model.Params.OutputFlag = 0
        model.Params.MIPFocus = 1
        model.Params.TimeLimit = solver_time_limit

        # Variables
        x = model.addVars(distances, obj=distances, vtype=gb.GRB.BINARY)
        if warm_start and iter > 0:
            if np.any(initial_labels == -1):
                raise ValueError(
                    "Warm start activated but labels are not fully initialized."
                )
            for (i, j), var in x.items():
                var.Start = 1 if initial_labels[i] == j else 0

        # Constraints
        model.addConstrs(x.sum(i, "*") == 1 for i in objects)
        model.addConstrs(x.sum("*", j) >= 1 for j in clusters)

        protected_group_labels = np.unique(self.sensitive_feature)
        for j in clusters:
            for g in protected_group_labels:
                for g_ in protected_group_labels:
                    if g != g_:
                        if self.algorithm == "mpfc":
                            count_g = x.sum(self.protected_groups_[g], j)
                            count_g_ = x.sum(self.protected_groups_[g_], j)
                            model.addConstr(
                                count_g >= self.target_balance_ * count_g_
                            )
                        elif self.algorithm == "smpfc":
                            counts_g = gb.quicksum(
                                self.batch_weights[i, g] * x[i, j]
                                for i in objects
                            )
                            counts_g_ = gb.quicksum(
                                self.batch_weights[i, g_] * x[i, j]
                                for i in objects
                            )
                            model.addConstr(
                                counts_g >= self.target_balance_ * counts_g_
                            )
        return model, x

    def _setup_blp_scip(
        self,
        distances: dict,
        initial_labels: np.ndarray,
        solver_time_limit: float,
        iter: int,
        warm_start: bool = False,
    ) -> tuple[scip.Model, dict]:
        """
        Build the BLP model for the assignment of objects to cluster centers
        using SCIP.

        Parameters
        ----------
        distances : dict
            {(i,j): distance from object i to cluster center j}.
        initial_labels : np.ndarray
            Cluster assignments from previous iteration (used for warm start).
        solver_time_limit : float
            SCIP solver time limit.
        warm_start : bool, default=False
            Boolean parameter to trigger warm start.

        Returns
        -------
        model : scip.Model
            Built SCIP model.
        x : dict
            Cluster assignment variables.
        """
        X = self.X if self.algorithm == "mpfc" else self.batch_X
        objects = range(X.shape[0])
        clusters = range(self.n_clusters)

        model = scip.Model()
        model.hideOutput()
        model.setEmphasis(scip.SCIP_PARAMEMPHASIS.FEASIBILITY)
        model.setParam("limits/time", solver_time_limit)

        # Variables
        x = {
            (i, j): model.addVar(vtype="B", obj=distances[i, j])
            for (i, j) in distances
        }
        if warm_start and iter > 0:
            if np.any(initial_labels == -1):
                raise ValueError(
                    "Warm start activated but labels are not fully initialized."
                )
            solution = model.createPartialSol()
            for (i, j), var in x.items():
                model.setSolVal(
                    solution, var, 1 if initial_labels[i] == j else 0
                )

        # Constraints
        for i in objects:
            model.addCons(scip.quicksum(x[i, j] for j in clusters) == 1)
        for j in clusters:
            model.addCons(scip.quicksum(x[i, j] for i in objects) >= 1)

        protected_group_labels = np.unique(self.sensitive_feature)
        for j in clusters:
            for g in protected_group_labels:
                for g_ in protected_group_labels:
                    if g != g_:
                        if self.algorithm == "mpfc":
                            count_g = scip.quicksum(
                                x[i, j] for i in self.protected_groups_[g]
                            )
                            count_g_ = scip.quicksum(
                                x[i, j] for i in self.protected_groups_[g_]
                            )
                            model.addCons(
                                count_g >= self.target_balance_ * count_g_
                            )
                        elif self.algorithm == "smpfc":
                            counts_g = scip.quicksum(
                                self.batch_weights[i, g] * x[i, j]
                                for i in objects
                            )
                            counts_g_ = scip.quicksum(
                                self.batch_weights[i, g_] * x[i, j]
                                for i in objects
                            )
                            model.addCons(
                                counts_g >= self.target_balance_ * counts_g_
                            )
        return model, x
