# MIT License
# Copyright (c) 2026 Claudio Mantuano, University of Bern
# Paper: https://arxiv.org/abs/2605.13759

import logging
import time

import faiss
import numpy as np
from ortools.graph.python.min_cost_flow import SimpleMinCostFlow

from fair_clustering.base import FairClustering


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


class FlowBasedHeuristic(FairClustering):
    """
    Subclass for the implementation of the heuristic based on minimum-cost flow (MCF).
    The algorithm alternates between a sequence of assignments through MCF (one for
    each protected group of the sensitive feature) and cluster center updates (with
    fixed assignments), according to the k-means decomposition scheme.

    Inherits all attributes from the base class FairClustering.

    Methods
    -------
    msflowfc()
        Solve fair k-means clustering using the Multi-Stage Minimum-Cost Flow-based Fair
        Clustering algorithm (MS-FlowFC).
    """

    def msflowfc(self) -> None:
        """Call this method to run the MS-FlowFC algorithm."""
        self.clustering_labels, self.status = self._run_decomposition_mcf()
        if self.status is None:
            self._extract_results()

    def _run_decomposition_mcf(
        self, max_iter: int = 100, min_improvement: float = 0.1
    ) -> tuple[np.ndarray | None, int | None]:
        """
        Implement the k-means decomposition scheme (initialization, assignment, and
        center update) to cluster objects until a stopping criterion is met (i.e.,
        the minimum cost improvement is not achieved, the time limit is exceeded, or
        the maximum number of iterations is attained). The assignment step consists
        of multiple stages. In the first stage, the objects from the largest protected 
        group are assigned to the closest cluster center using FAISS. In the following 
        stages, the objects from the remaining protected groups are assigned solving a 
        sequence of MCF problems.

        Parameters
        ----------
        max_iter : int, default=100
            Maximum number of iterations.
        min_improvement : float, default=0.1
            Minimum cost improvement (%) required to continue.

        Returns
        -------
        best_labels : np.ndarray | None
            Best assignments, or -1 array if no solution.
        status : int | None
            Returns status if failed, None otherwise.
        """
        best_labels = np.full(self.X.shape[0], -1, dtype=np.int32)
        best_cost = float("inf")
        start_time = time.perf_counter()

        centers = self._initialize_centers_kmeans_pp(
            X=self.X[self.protected_groups[0]],
            n_centers=self.n_clusters,
            seed=self.seed,
        )

        while self.n_iter < max_iter:
            empty_clusters = np.arange(self.n_clusters)
            labels = np.full(self.X.shape[0], -1, dtype=np.int32)
            assignment_start_time = time.perf_counter()

            # Assignment of first protected group is repeated until no cluster is empty
            while empty_clusters.shape[0] > 0:
                # First protected group assignment using FAISS
                first_stage_labels = self._assign_objects_faiss(
                    objects=self.X[self.protected_groups[0]],
                    centers=centers,
                )
                labels[self.protected_groups[0]] = first_stage_labels

                # Check cluster emptiness
                non_empty_clusters = np.unique(
                    labels[self.protected_groups[0]]
                )
                clusters = np.arange(self.n_clusters)
                empty_clusters = np.setdiff1d(clusters, non_empty_clusters)

                if empty_clusters.shape[0] > 0:
                    # Seed is updated to generate new cluster centers
                    self.seed += 1
                    # Center of empty clusters is re-initialized using k-means++
                    centers[empty_clusters] = (
                        self._initialize_centers_kmeans_pp(
                            X=self.X[self.protected_groups[0]],
                            n_centers=empty_clusters.shape[0],
                            seed=self.seed,
                        )
                    )

            # Assignment of objects from remaining protected groups
            idx_assigned_groups = [0]
            for g in range(1, len(self.protected_groups)):
                # Count objects from most represented group per cluster
                counts = self._get_cluster_representation(
                    labels=labels, idx_assigned_groups=idx_assigned_groups
                )
                # MCF-based assignment
                multi_stage_labels, _ = self._assign_objects_min_cost_flow(
                    current_group=g,
                    objects=self.X[self.protected_groups[g]],
                    centers=centers,
                    counts=counts,
                )
                labels[self.protected_groups[g]] = multi_stage_labels
                idx_assigned_groups.append(g)  # add last protected group
            assignment_runtime = time.perf_counter() - assignment_start_time

            # Update step
            centers, update_runtime = self._update_centers(
                X=self.X, labels=labels, n_clusters=self.n_clusters
            )
            cost = self._get_cost(X=self.X, centers=centers, labels=labels)

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
                    self.clustering_cost = None
                    self.runtime = time.perf_counter() - start_time
                    return best_labels, 9
                else:
                    break
            else:
                best_cost = cost
                best_labels = labels.copy()

            logger.info(
                "iter=%-2d update[s]=%6.6f  assignment[s]=%6.4f  cost=%6.4f",
                self.n_iter,
                update_runtime,
                assignment_runtime,
                cost,
            )
            self.n_iter += 1

        self.runtime = time.perf_counter() - start_time
        return best_labels, None

    def _get_cluster_representation(
        self, labels: np.ndarray, idx_assigned_groups: list
    ) -> dict:
        """
        Compute maximum and minimum protected group counts per cluster.

        Parameters
        ----------
        labels : np.ndarray
            Cluster assignments.
        idx_assigned_groups : list
            List containing indices of already assigned protected groups.

        Returns
        -------
        representation : dict
            Dictionary with 'max' and 'min' arrays containing maximum and minimum
            protected group counts for each cluster, respectively.
        """
        clusters = range(self.n_clusters)
        representation = {
            "max": np.zeros(self.n_clusters, dtype=int),
            "min": np.zeros(self.n_clusters, dtype=int),
        }

        for cluster in clusters:
            protected_group_counts = np.array([
                np.sum(labels[self.protected_groups[g]] == cluster)
                for g in idx_assigned_groups
            ])
            representation["max"][cluster] = protected_group_counts.max()
            representation["min"][cluster] = protected_group_counts.min()
        return representation

    def _assign_objects_min_cost_flow(
        self,
        current_group: int,
        objects: np.ndarray,
        centers: np.ndarray,
        counts: dict,
    ) -> tuple[np.ndarray, float]:
        """
        Assign objects using MCF with fairness constraints enforced by node
        demands and arc capacities.

        Parameters
        ----------
        current_group : int
            Label of protected group being assigned.
        objects : np.ndarray
            Non-sensitive feature matrix of objects being assigned.
        centers : np.ndarray
            Coordinate array of cluster centers (fixed).
        counts : dict
            Protected group counts per cluster.

        Returns
        -------
        labels : np.ndarray
            Cluster assignments of objects from the current protected group.
        elapsed_time : float
            Elapsed time for MCF optimization.
        """
        n_objects = objects.shape[0]
        sink = n_objects + self.n_clusters
        distances = self._compute_distance_matrix(objects, centers)
        start_time = time.perf_counter()

        # Lower bounds
        demand = self._compute_demand(
            current_group=current_group, counts=counts["max"]
        )
        # Upper bounds
        capacity = self._compute_capacity(
            current_group=current_group, counts=counts["min"], demand=demand
        )

        model, status = self._construct_mcf_network(
            distances=distances,
            demand=demand,
            capacity=capacity,
        )
        elapsed_time = time.perf_counter() - start_time

        if status != model.OPTIMAL:
            raise RuntimeError("The minimum-cost flow problem is infeasible.")

        mcf_labels = np.full(n_objects, -1, dtype=np.int32)
        arcs = range(model.num_arcs())
        for arc in arcs:
            tail = model.tail(arc)
            head = model.head(arc)
            flow = model.flow(arc)
            if 0 <= tail < n_objects and n_objects <= head < sink and flow > 0:
                mcf_labels[tail] = head - n_objects
        return mcf_labels, elapsed_time

    def _compute_demand(
        self, current_group: int, counts: np.ndarray
    ) -> np.ndarray:
        """
        Compute demand to ensure that a sufficient number of objects is assigned
        to each cluster from the current protected group to satisfy fairness.

        Parameters
        ----------
        current_group : int
            Label of protected group being assigned.
        counts : np.ndarray
            Current object counts per cluster.

        Returns
        -------
        demand : np.ndarray
            Demand vector (negative values).
        """
        n_objects_group = self.protected_groups[current_group].shape[0]
        # Raw demand (negative values)
        demand = -np.ceil(self.target_balance * counts).astype(np.int32)
        # Demand cannot be positive
        demand[demand > 0] = 0
        # Residual demand to pass to the sink
        gap = -n_objects_group - np.sum(demand)
        # If residual demand is positive due to rounding, adjust
        if gap > 0:
            demand = self._adjust_demand(
                demand_gap=gap, demand=demand, counts=counts
            )
        return demand

    def _compute_capacity(
        self, current_group: int, counts: np.ndarray, demand: np.ndarray
    ) -> np.ndarray:
        """
        Compute capacity to bound the number of objects from the current protected
        group assigned to each cluster to satisfy fairness.

        Parameters
        ----------
        current_group : int
            Label of protected group being assigned.
        counts : np.ndarray
            Current object counts per cluster.
        demand : np.ndarray
            Demand vector (negative values).

        Returns
        -------
        capacity : np.ndarray
            Capacity vector (non-negative values).
        """
        n_objects_group = self.protected_groups[current_group].shape[0]
        # Raw capacity (non-negative values)
        capacity = np.floor((1 / self.target_balance) * counts).astype(
            np.int32
        )
        capacity += demand
        # Capacity cannot be negative
        capacity[capacity < 0] = 0
        # If total capacity is insufficient due to rounding, adjust
        total_capacity = np.sum(capacity)
        total_demand = np.sum(demand)
        gap = (n_objects_group + total_demand) - total_capacity
        if gap > 0:
            capacity = self._adjust_capacity(
                capacity_gap=gap,
                demand=demand,
                capacity=capacity,
                counts=counts,
            )
        return capacity

    def _construct_mcf_network(
        self,
        distances: np.ndarray,
        demand: np.ndarray,
        capacity: np.ndarray,
        C: float = 1e3,
    ) -> tuple[SimpleMinCostFlow, SimpleMinCostFlow.Status]:
        """
        Construct the MCF network.

        Parameters
        ----------
        distances : np.ndarray
            Matrix of distance from each object to each cluster center.
        demand : np.ndarray
            Demand vector per cluster (lower bound).
        capacity : np.ndarray
            Capacity vector per cluster (upper bound).
        C : float
            Scaling factor used for cost (distance) conversion from float to int.

        Returns
        -------
        model : SimpleMinCostFlow
            Solved MCF model.
        status : SimpleMinCostFlow.Status
            MCF solution status.
        """
        n_objects = distances.shape[0]
        objects = range(n_objects)
        clusters = range(self.n_clusters)
        sink = n_objects + self.n_clusters

        mcf = SimpleMinCostFlow()
        # Transform distances (costs) to integers for OR-Tools
        distances_mcf = np.round(distances * C).astype(np.int32)
        # Arcs from objects to cluster centers
        for obj in objects:
            for cluster in clusters:
                mcf.add_arc_with_capacity_and_unit_cost(
                    tail=obj,
                    head=n_objects + cluster,
                    capacity=1,
                    unit_cost=distances_mcf[obj, cluster],
                )
        # Arcs from cluster centers to sink
        for cluster in clusters:
            mcf.add_arc_with_capacity_and_unit_cost(
                tail=n_objects + cluster,
                head=sink,
                capacity=capacity[cluster].astype(np.int32),
                unit_cost=0,
            )
        # Object supply
        for obj in objects:
            mcf.set_nodes_supplies(nodes=obj, supplies=1)
        # Cluster center demand
        for cluster in range(n_objects, sink):
            cluster_idx = cluster - n_objects
            mcf.set_nodes_supplies(nodes=cluster, supplies=demand[cluster_idx])
        # Sink demand
        mcf.set_nodes_supplies(
            nodes=sink, supplies=-n_objects - np.sum(demand)
        )
        status = mcf.solve()
        return mcf, status

    @staticmethod
    def _assign_objects_faiss(
        objects: np.ndarray, centers: np.ndarray
    ) -> np.ndarray:
        """
        Assign objects to nearest cluster centers using FAISS L2 distance.

        Parameters
        ----------
        objects : np.ndarray
            Non-sensitive feature matrix of objects being assigned.
        centers : np.ndarray
            Coordinate array of cluster centers (fixed).

        Returns
        -------
        labels : np.ndarray
            Cluster assignments.
        """
        d = centers.shape[1]
        # FAISS requires float32
        objects = np.asarray(objects, dtype=np.float32)
        centers = np.asarray(centers, dtype=np.float32)
        index = faiss.IndexFlatL2(d)
        index.add(centers)
        _, faiss_labels = index.search(objects, 1)
        labels = faiss_labels[:, 0].astype(int)
        return labels

    @staticmethod
    def _adjust_demand(
        demand_gap: int, demand: np.ndarray, counts: np.ndarray
    ) -> np.ndarray:
        """
        Adjust demand to ensure MCF feasibility by greedily reducing demand in 
        clusters with the highest expected balance after adjustment.

        Parameters
        ----------
        demand_gap : int
            Units of excess demand.
        demand : np.ndarray
            Current demand vector.
        counts : np.ndarray
            Current object counts per cluster.

        Returns
        -------
        adj_demand : np.ndarray
            Adjusted demand vector.
        """
        adj_demand = demand.copy()
        while demand_gap > 0:
            # Expected balance of each cluster if demand is reduced by 1
            expected_balance = -(adj_demand + 1) / counts
            # Sort clusters by highest expected balance
            sorted_clusters = np.argsort(expected_balance)[::-1]
            # Reduce demand in best cluster (with the highest expected balance)
            for idx in sorted_clusters:
                if adj_demand[idx] < 0:
                    adj_demand[idx] += 1
                    demand_gap -= 1
                    break
        return adj_demand

    @staticmethod
    def _adjust_capacity(
        capacity_gap: int,
        demand: np.ndarray,
        capacity: np.ndarray,
        counts: np.ndarray,
    ) -> np.ndarray:
        """
        Adjust capacity to ensure MCF feasibility by greedily increasing capacity 
        in clusters with the highest expected balance after adjustment.

        Parameters
        ----------
        capacity_gap : int
            Units of additional capacity required.
        capacity : np.ndarray
            Current capacity vector.
        counts : np.ndarray
            Current object counts per cluster.

        Returns
        -------
        adj_capacity : np.ndarray
            Adjusted capacity vector.
        """
        adj_capacity = capacity.copy()
        while capacity_gap > 0:
            # Expected balance of each cluster if capacity is increased by 1
            expected_balance = counts / (adj_capacity - demand + 1)
            # Increase capacity in the cluster with highest expected balance
            idx = np.argmax(expected_balance)
            adj_capacity[idx] += 1
            capacity_gap -= 1
        return adj_capacity
