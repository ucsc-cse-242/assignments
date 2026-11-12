"""Question 3: noisy-label SVM and K-NN repair (40 points).

Complete the TODOs. Keep test labels clean and use them only for evaluation.
"""

import argparse
from pathlib import Path

SVM_C = 1.0
SVM_GAMMA = 0.01
POSITIVE_FLIP_PROBABILITY = 0.35
NEGATIVE_FLIP_PROBABILITY = 0.20


def generate_clean_split(rng, samples_per_class=1000):
    """Return features and labels for one independent clean split.

    Positive class: mean (1, 1), covariance I.
    Negative class: mean (-1, -1), covariance 3I.
    The negative covariance entries are variances, not standard deviations.
    """
    import numpy as np
    positive = rng.multivariate_normal((1, 1), np.eye(2), samples_per_class)
    negative = rng.multivariate_normal((-1, -1), 3 * np.eye(2), samples_per_class)
    features = np.vstack((positive, negative))
    labels = np.concatenate((np.ones(samples_per_class, dtype=int),
                             -np.ones(samples_per_class, dtype=int)))
    return features, labels


def corrupt_training_labels(clean_labels, rng):
    """Independently flip +1 with probability .35 and -1 with probability .20.

    Flip probabilities are chosen from the ORIGINAL labels, and exactly one
    uniform draw is made per example, so nothing can be flipped twice.
    Returns a new array; clean_labels is unchanged.
    """
    import numpy as np
    clean_labels = np.asarray(clean_labels)
    p_flip = np.where(clean_labels == 1,
                      POSITIVE_FLIP_PROBABILITY, NEGATIVE_FLIP_PROBABILITY)
    flips = rng.random(len(clean_labels)) < p_flip
    return np.where(flips, -clean_labels, clean_labels)


def repair_labels_knn(train_features, noisy_labels, neighbors):
    """Q3(b): K-NN majority-vote label repair; returns repaired labels.

    Uses only training features and noisy labels. Each point's own index is
    excluded from its neighborhood; all repairs read the ORIGINAL noisy array.
    Equal-distance ties are resolved by the library consistently (the smaller
    row index comes first after a stable ordering of the returned neighbors).
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    noisy_labels = np.asarray(noisy_labels)
    nn = NearestNeighbors(n_neighbors=neighbors + 1, algorithm="brute",
                          metric="euclidean").fit(train_features)
    dist, idx = nn.kneighbors(train_features)
    n = len(train_features)
    neigh = np.empty((n, neighbors), dtype=int)
    for i in range(n):
        # stable sort by (distance, index) for a consistent tie rule
        order = np.lexsort((idx[i], dist[i]))
        row = idx[i][order]
        row = row[row != i]            # exclude the point itself
        neigh[i] = row[:neighbors]
    votes = noisy_labels[neigh].sum(axis=1)     # five +/-1 votes: never 0
    return np.where(votes > 0, 1, -1)


def train_and_evaluate(train_features, train_labels, test_features, test_labels):
    """Train the required RBF SVM and return accuracy on the clean test set."""
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    model = SVC(kernel="rbf", C=SVM_C, gamma=SVM_GAMMA)
    model.fit(train_features, train_labels)
    return float(accuracy_score(test_labels, model.predict(test_features)))


def run_comparison(repeats, seed, neighbors, output_path):
    """Q3(a-b): compare the noisy baseline with K-NN label repair."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    train_x, train_y = generate_clean_split(rng)    # fixed across trials
    test_x, test_y = generate_clean_split(rng)      # independent, clean

    rows = []
    for trial in range(1, repeats + 1):
        noisy_y = corrupt_training_labels(train_y, rng)
        baseline = train_and_evaluate(train_x, noisy_y, test_x, test_y)
        repaired_y = repair_labels_knn(train_x, noisy_y, neighbors)
        knn = train_and_evaluate(train_x, repaired_y, test_x, test_y)
        rows.append({"trial": trial, "baseline_accuracy": baseline,
                     "knn_accuracy": knn})

    df = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    mb, mk = df.baseline_accuracy.mean(), df.knn_accuracy.mean()
    print(f"Baseline mean accuracy: {mb:.4f}")
    print(f"K-NN repair mean accuracy: {mk:.4f}")
    print(f"Difference (K-NN - baseline): {mk - mb:+.4f}")
    print(f"Saved per-trial results to {output_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--neighbors", type=int, default=5,
                        choices=(5,), help="Use five neighbors for this assignment.")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "results" / "noise_comparison.csv")
    args = parser.parse_args()
    run_comparison(args.repeats, args.seed, args.neighbors, args.output)


if __name__ == "__main__":
    main()
