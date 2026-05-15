# MIT License
# Copyright (c) 2026 Claudio Mantuano, University of Bern
# Paper: https://arxiv.org/abs/2605.13759

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_clustering(
    dataset: str,
    X: np.ndarray,
    sensitive_name: str,
    sensitive_feature: np.ndarray,
    protected_groups: list[str],
    algorithm: str,
    k: int,
    tolerance: float,
    labels: np.ndarray,
    centers: np.ndarray,
    cost: float,
    balance: float,
    cluster_balances: np.ndarray,
) -> None:
    """Plot clustering in two dimensions."""
    if X.shape[1] != 2:
        raise ValueError(
            f"plot_clustering supports only 2D data, got {X.shape[1]} features."
        )

    colors = ["green", "blue", "red", "orange", "gray"]
    if len(colors) < len(protected_groups):
        raise ValueError(
            f"{len(colors)} colors available for {len(protected_groups)} protected groups"
        )
    group_colors = colors[: len(protected_groups)]

    fig, ax = plt.subplots(figsize=(6, 6))
    for group_id, color in enumerate(group_colors):
        mask = sensitive_feature == group_id
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            s=100,
            alpha=0.5,
            c=color,
            label=f"{protected_groups[group_id]}",
        )
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        s=50,
        c="black",
        marker="x",
        linewidths=2,
        label="Centers",
    )
    for cluster_id in np.unique(labels):
        cluster_points = X[labels == cluster_id]
        center_x, center_y = centers[cluster_id]
        for point_x, point_y in cluster_points:
            ax.plot(
                [point_x, center_x],
                [point_y, center_y],
                linestyle="--",
                linewidth=0.5,
                color="black",
            )
    for cluster_id, bal in enumerate(cluster_balances):
        center_x, center_y = centers[cluster_id]
        ax.text(
            center_x + 0.05,
            center_y - 0.05,
            f"{bal:.2f}",
            fontsize=12,
            color="black",
            bbox=dict(
                facecolor="white",
                edgecolor="black",
                boxstyle="square,pad=0.2",
            ),
        )
    ax.set_title(
        f"Dataset: {dataset}, Sensitive feature: {sensitive_name}\n"
        f"Algorithm: {algorithm}, k: {k}, Tolerance: {tolerance}\n"
        f"Cost = {cost:.2f}, Balance = {balance:.2f}",
        fontsize="x-large",
    )
    ax.tick_params(axis="both", labelsize="x-large")
    ax.legend(loc="upper left", fontsize="x-large")
    ax.grid(True, linestyle="--", alpha=0.2)

    Path("results").mkdir(parents=True, exist_ok=True)
    filename = f"results/{dataset}_{algorithm}_k{k}_tol{tolerance}.png"
    fig.savefig(filename, dpi=300)
    plt.close(fig)
