"""Question 2: K-means (30 points), adapted from Assignment 4, Question 2.

Implement the two update functions from scratch, then complete the experiment.
"""

import argparse
from pathlib import Path
import numpy as np

K_VALUES = (2, 3, 4, 6, 10)


def update_assignments(data, centroids):
    """Return one cluster index per row, using nearest-centroid distance.

    Q2(a): the two update functions together are worth 15 points.
    data has shape (n, d); centroids has shape (k, d).
    The result should be an integer array with shape (n,).
    Resolve equal-distance ties consistently and document your rule.
    """
    # TODO: Compute distances and assign each point to a centroid.
    # for ties, if a point is equally close to multiple centroids
    # it goes to the lowest index because its what np.argmin returns
    # this also allows repeatable results as it is deterministic

    # for empty clusters, update_centroids will keep previous position
    # of a centroid that received no points to avoid any issues when averaging
    # no points, it also works on a copy
    data = np.asarray(data, dtype = float)
    centroids = np.asarray(centroids, dtype = float)
    num_points, num_centroids = len(data), len(centroids)
    squared_distances = np.empty((num_points, num_centroids))
    for i, centroid in enumerate(centroids):
        # squared distance from every point to centroid i
        squared_distances[:, i] = ((data - centroid) ** 2).sum(axis = 1)
    # each row holds a points distance to each centroid, pick smallest
    return np.argmin(squared_distances, axis = 1)
    raise NotImplementedError("Implement the assignment step from scratch.")


def update_centroids(data, centroids, assignments):
    """Return an updated (k, d) centroid array based on cluster assignments.

    Document how you handle an empty cluster. One valid choice is to retain
    its previous centroid. Avoid modifying the input centroid array in place.
    """
    # TODO: Update the centroid of each nonempty cluster.
    # TODO: Handle empty clusters using your documented policy.
    # a centroid receiving no points keeps previous position
    data = np.asarray(data, dtype = float)
    # avoid touching inputs
    new_centroids = np.array(centroids, dtype = float, copy = True)
    for i in range(new_centroids.shape[0]):
        points = data[assignments == i]
        if len(points) > 0:
            new_centroids[i] = points.mean(axis = 0)
    return new_centroids
    raise NotImplementedError("Implement the centroid step from scratch.")


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
    """Q2(b), 15 points: run all five k values and save cluster plots.

    Use the four-component truth labels only for a reference plot. K-means
    must cluster the features without access to those labels.
    """
    # TODO: Load toy data, create a seeded RNG, and initialize centroids.
    # TODO: Run kmeans for every value in K_VALUES.
    # TODO: Plot each set of cluster assignments and its final centroids.
    # TODO: Save plots under output_dir and discuss the effect of k.
    import matplotlib
    import matplotlib.pyplot as plt 

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data, true_labels = make_toy_data()
    rng = np.random.default_rng(seed)

    # reference plot
    fig, ax = plt.subplots(figsize = (6, 5))
    ax.scatter(data[:, 0], data[:, 1], c = true_labels, s = 6, cmap = "tab10")
    ax.set(title = "Reference", xlabel = "x1", ylabel = "x2")
    fig.tight_layout()
    fig.savefig(output_dir / "reference_groups.png", dpi = 150)
    plt.close(fig)

    results = {}
    for k in K_VALUES:
        initial = get_initial_clusters(k, data, rng)
        centroids, assignments = kmeans(data, initial, max_iterations)
        # sum of squared within each cluster
        cluster_square_sums = float(((data - centroids[assignments]) ** 2).sum())
        nonempty = len(np.unique(assignments))
        results[k] = (cluster_square_sums, nonempty)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(data[:, 0], data[:, 1], c=assignments, s=6,
                   cmap="tab10", vmin=0, vmax=9)
        ax.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=180,
                   c="black", edgecolors="white", 
                   label="final centroids")
        ax.set(title=f"K-means, k = {k}",
               xlabel="x1", ylabel="x2")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output_dir / f"kmeans_k{k}.png", dpi=150)
        plt.close(fig)
    
    return results
    raise NotImplementedError("Complete the K-means experiment and plots.")
    # as we increase the k, the clusters get smaller and centroids sit
    # closer to the points, but the meaning within each cluster changes
    # when k=2 the top cluster is isolated and the 3 other groups bunch 
    # together to one group, when k=3 the top and bottom cluster isolates
    # but the middle two bunch together, at k=4 it resembles the reference
    # the most matching each cluster, for k=6 and k=10 it requires a use
    # of extra centroids, so it splits real groups into smaller clusters
    # so while a larger k fits the data more tightly, there needs to be a 
    # balance and analysis as to what value of k fits the data most effectively
    # low k merges distinct groups and high k fragments them


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
