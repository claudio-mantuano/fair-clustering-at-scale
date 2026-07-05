# MIT License
# Copyright (c) 2026 Claudio Mantuano, University of Bern
# Paper: https://arxiv.org/abs/2605.13759

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import gurobipy as gb


class ExactApproaches:
    """
    Class implementing exact approaches for fair k-means clustering using
    an MIQCP formulation implemented in Gurobi or a Set-Variable-based
    (SetVars) formulation implemented in Hexaly.

    Provides the algorithms "miqcp" and "setvars" to the FairClustering class
    defined in fair_clustering.base, which invokes them through `fit()`.

    Methods
    -------
    _miqcp()
        Solve fair k-means clustering using MIQCP with Gurobi.
    _setvars()
        Solve fair k-means clustering using SetVars with Hexaly.
    """

    def _miqcp(self) -> None:
        """Run the MIQCP approach."""
        model, labels = self._build_miqcp_model_gurobi()
        start_time = time.perf_counter()
        model.optimize()
        self.runtime_ = time.perf_counter() - start_time
        self._extract_results_gurobi(model, labels)

    def _setvars(self) -> None:
        """Run the SetVars approach."""
        optimizer, clusters = self._build_setvars_model_hexaly()
        start_time = time.perf_counter()
        optimizer.solve()
        self.runtime_ = time.perf_counter() - start_time
        self._extract_results_hexaly(optimizer, clusters)
        self._extract_results()
        optimizer.delete()  # release Hexaly license token

    def _extract_results_gurobi(
        self, miqcp_model: gb.Model, labels: gb.tupledict
    ) -> None:
        """
        Extract results from solved Gurobi model.

        Parameters
        ----------
        miqcp_model : gb.Model
            Solved Gurobi model.
        labels : gb.tupledict
            Solved Gurobi model's assignment variables.
        """
        n_objects = range(self.X.shape[0])
        n_clusters = range(self.n_clusters)
        if miqcp_model.SolCount > 0:
            self.labels_ = np.array(
                [
                    j
                    for i in n_objects
                    for j in n_clusters
                    if labels[i, j].X > 0.5
                ],
                dtype=int,
            )
            self.mipgap_ = miqcp_model.MIPGap
            self._extract_results()
        else:
            self.cost_ = None
            self.status_ = miqcp_model.Status

    def _extract_results_hexaly(self, optimizer, clusters) -> None:
        """Extract results from solved Hexaly model."""
        for cluster_id, cluster in enumerate(clusters):
            self.labels_[cluster.value] = cluster_id
        self.mipgap_ = optimizer.solution.get_objective_gap(pos=0)

    def _build_miqcp_model_gurobi(self) -> tuple[gb.Model, gb.tupledict]:
        """Construct MIQCP model for Gurobi."""
        try:
            import gurobipy as gb
        except ImportError as e:
            raise ImportError(
                "The 'miqcp' algorithm requires gurobipy, which is not installed. "
                "Install the gurobipy version matching your local Gurobi installation."
            ) from e
        
        n, d = self.X.shape
        objects = range(n)
        clusters = range(self.n_clusters)
        features = range(d)
        distance_matrix = self._compute_distance_matrix(self.X, self.X)
        big_M = np.max(distance_matrix)
        model = self._setup_solver(solver="gurobi")

        mu = model.addVars(
            clusters,
            features,
            lb={
                (j, f): np.min(self.X[:, f])
                for j in clusters
                for f in features
            },
            ub={
                (j, f): np.max(self.X[:, f])
                for j in clusters
                for f in features
            },
        )
        x = model.addVars(objects, clusters, vtype=gb.GRB.BINARY)
        distances = model.addVars(objects, clusters, ub=big_M)
        model.update()

        model.addConstrs(x.sum(i, "*") == 1 for i in objects)
        model.addConstrs(x.sum("*", j) >= 1 for j in clusters)
        model.addConstrs(
            (
                gb.quicksum((self.X[i, f] - mu[j, f]) ** 2 for f in features)
                <= distances[i, j] + big_M * (1 - x[i, j])
                for i in objects
                for j in clusters
            )
        )

        protected_group_labels = np.unique(self.sensitive_feature)
        for j in clusters:
            for g in protected_group_labels:
                for g_ in protected_group_labels:
                    if g != g_:
                        counts_g = x.sum(self.protected_groups_[g], j)
                        counts_g_ = x.sum(self.protected_groups_[g_], j)
                        model.addConstr(
                            counts_g >= self.target_balance_ * counts_g_
                        )

        model.setObjective(distances.sum(), gb.GRB.MINIMIZE)
        model.update()
        return model, x

    def _build_setvars_model_hexaly(self):
        """Construct set variable model for Hexaly."""
        n_objects, n_features = self.X.shape
        clusters = range(self.n_clusters)
        features = range(n_features)
        optimizer, model, relative_gap = self._setup_solver(solver="hexaly")
        X = model.array(self.X)  # convert NumPy arrays to Hexaly arrays
        sensitive_feature = model.array(self.sensitive_feature)

        cluster_set_vars = [model.set(n_objects) for _ in clusters]
        model.constraint(model.partition(cluster_set_vars))
        clustering_cost = []
        for cluster in cluster_set_vars:
            protected_group_labels = np.unique(self.sensitive_feature)
            for g in protected_group_labels:
                for g_ in protected_group_labels:
                    if g != g_:
                        counts_g = model.sum(
                            cluster,
                            model.lambda_function(
                                lambda i: model.iif(
                                    model.at(sensitive_feature, i) == g, 1, 0
                                )
                            ),
                        )
                        counts_g_ = model.sum(
                            cluster,
                            model.lambda_function(
                                lambda i: model.iif(
                                    model.at(sensitive_feature, i) == g_, 1, 0
                                )
                            ),
                        )
                        model.constraint(
                            counts_g >= self.target_balance_ * counts_g_
                        )
            size = model.count(cluster)
            centers = []
            for f in features:
                coordinate_lambda = model.lambda_function(
                    lambda i: model.at(X, i, f)
                )
                coordinate_f = model.iif(
                    size == 0, 0, model.sum(cluster, coordinate_lambda) / size
                )
                centers.append(coordinate_f)

            cluster_cost = model.sum()
            for f in features:
                dimension_variance_lambda = model.lambda_function(
                    lambda i: model.sum(
                        model.pow(model.at(X, i, f) - centers[f], 2)
                    )
                )
                dimension_cost = model.sum(cluster, dimension_variance_lambda)
                cluster_cost.add_operand(dimension_cost)
            clustering_cost.append(cluster_cost)

        objective = model.sum(clustering_cost)
        model.minimize(objective)
        model.close()
        optimizer.param.set_objective_threshold(0, relative_gap)
        return optimizer, cluster_set_vars

    def _setup_solver(self, solver: str, relative_gap: float = 0.0):
        """Initialize and configure optimization solver (Gurobi or Hexaly)."""
        if solver == "gurobi":
            import gurobipy as gb

            model = gb.Model()
            # MIPFocus: 0=balanced, 1=feasibility, 2=optimality, 3=bound
            model.Params.MIPFocus = 0
            model.Params.OutputFlag = 1  # 0=silent, 1=normal logging
            model.Params.MIPGap = relative_gap  # 0.0 = exact optimum
            model.Params.TimeLimit = self.time_limit
            return model
        elif solver == "hexaly":
            try:
                import hexaly.optimizer as hx
            except ImportError as e:
                raise ImportError(
                    "The 'setvars' algorithm requires hexaly, which is not installed. " 
                    "Install the hexaly version matching your local Hexaly installation "
                    "(https://www.hexaly.com/docs/last/installation/pythonsetup.html)."
                ) from e
            
            optimizer = hx.HexalyOptimizer()
            model = optimizer.model
            optimizer.param.verbosity = 1  # 0=quiet, 1=normal, 2=detailed
            optimizer.param.time_limit = self.time_limit
            # Note: Hexaly API requires model to be closed before setting MIP Gap
            return optimizer, model, relative_gap
        else:
            raise ValueError(
                "Please select a supported solver ('gurobi', 'hexaly')."
            )
