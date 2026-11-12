"""Question 2: K-means (30 points), adapted from Assignment 4, Question 2.

Implement the two update functions from scratch, then complete the experiment.
"""

import argparse
from pathlib import Path

K_VALUES = (2, 3, 4, 6, 10)


def update_assignments(data, centroids):
    """Return one cluster index per row, using nearest-centroid distance.

    Tie rule: a point equidistant from several centroids goes to the one with
    the LOWEST index (np.argmin returns the first minimum).
    """
    import numpy as np
    # squared Euclidean distances, shape (n, k)
    d2 = ((data[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(d2, axis=1).astype(int)


def update_centroids(data, centroids, assignments):
    """Return an updated (k, d) centroid array based on cluster assignments.

    Empty-cluster policy: an empty cluster's centroid is re-seeded to the data
    point currently farthest from its own assigned centroid (distinct points for
    distinct empty clusters). This also rescues duplicate initial centroids,
    where the higher-index duplicate always loses ties and would stay empty.
    The input centroid array is not modified.
    """
    import numpy as np
    new = np.array(centroids, dtype=float, copy=True)
    empty = []
    for j in range(len(centroids)):
        mask = assignments == j
        if mask.any():
            new[j] = data[mask].mean(axis=0)
        else:
            empty.append(j)
    if empty:
        d2_own = ((data - centroids[assignments]) ** 2).sum(axis=1)
        farthest_first = np.argsort(-d2_own, kind="stable")
        for j, idx in zip(empty, farthest_first):
            new[j] = data[idx]
    return new


def kmeans(data, centroids, max_iterations=100):
    """Assignment 4's provided alternating-update loop, with a final assignment."""
    for _ in range(max_iterations):
        assignments = update_assignments(data, centroids)
        centroids = update_centroids(data, centroids, assignments)
    assignments = update_assignments(data, centroids)
    return centroids, assignments


def make_toy_data():
    """Generate the exact make_blobs configuration from Assignment 4."""
    from sklearn.datasets import make_blobs
    return make_blobs(n_samples=4000, centers=4, cluster_std=0.60, random_state=0)


def get_initial_clusters(k, data, rng):
    """Sample k rows with replacement, as in the source's randint initializer.

    This adaptation accepts a seeded NumPy generator for reproducibility.
    Repeated initial rows are possible; your empty-cluster policy must cope.
    """
    return data[rng.integers(0, data.shape[0], size=k)].copy()


def run_experiments(output_dir, seed=0, max_iterations=100):
    """Q2(b): run all five k values and save cluster plots.

    The truth labels are used only for a reference plot.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data, truth = make_toy_data()
    rng = np.random.default_rng(seed)

    # Reference plot (truth labels used for display only).
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(data[:, 0], data[:, 1], c=truth, s=4, cmap="tab10")
    ax.set_title("Reference: four generating groups")
    fig.savefig(output_dir / "reference_truth.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    results = {}
    for k in K_VALUES:
        init = get_initial_clusters(k, data, rng)
        centroids, assignments = kmeans(data, init, max_iterations)
        inertia = float(((data - centroids[assignments]) ** 2).sum())
        results[k] = (centroids, assignments, inertia)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(data[:, 0], data[:, 1], c=assignments, s=4, cmap="tab10")
        ax.scatter(centroids[:, 0], centroids[:, 1], c="black", marker="X",
                   s=160, edgecolors="white", label="centroids")
        ax.set_title(f"K-means, k={k} (inertia={inertia:.0f})")
        ax.legend()
        fig.savefig(output_dir / f"kmeans_k{k}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"k={k:2d}: inertia={inertia:10.1f}, "
              f"non-empty clusters={len(np.unique(assignments))}")

    print("""
Discussion: with k=2 and k=3 the algorithm must merge the four generating
groups (large inertia). At k=4 each centroid sits in one blob and inertia has
an elbow (a poor random init can still split one blob and merge two). For
k=6 and k=10 blobs are over-segmented into arbitrary sub-clusters through
dense Gaussian cores; inertia keeps falling but the extra clusters reflect
no real structure.""")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results" / "kmeans")
    args = parser.parse_args()
    run_experiments(args.output_dir, args.seed, args.max_iterations)


if __name__ == "__main__":
    main()
