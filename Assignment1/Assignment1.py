"""CSE 242 Assignment 1 — submit as Assignment1.py.

Fill # YOUR ANSWER HERE blocks with commented reasoning; fill function bodies
at # YOUR CODE HERE. Keep signatures. See README.md and INTERFACE.md.
Functions must not mutate inputs, prompt for input, or perform network I/O.
Only code inside the main guard should save/show figures.
"""
import numpy as np
import math
import matplotlib.pyplot as plt

# PART 1 (20 points), Q1: Show computations for the given A and B.
# YOUR ANSWER HERE
def matrix_operations(A=None, B=None):
    """Return AB, BA, (A+B).T, A@B.T, trace(A), trace(AB).

    Defaults: A=[[3,4],[2,1]], B=[[1,2],[0,-1]]. Other inputs are square
    real matrices of matching size. Return a tuple in the specified order.
    """
    # YOUR CODE HERE
    if A is None:
        A=[[3,4],[2,1]]
    if B is None:
        B=[[1,2],[0,-1]]
    
    A = np.array(A)
    B = np.array(B)

    AB = A @ B
    BA = B @ A 
    AplusB_T = (A + B).T
    AB_T = A @ B.T 
    traceA = np.trace(A)
    traceAB = np.trace(AB)

    return (AB, BA, AplusB_T, AB_T, traceA, traceAB)

# print(matrix_operations())


# P1 Q2: Derive the given M eigenpairs and verify Mv=lambda*v.
# YOUR ANSWER HERE
def eigensystem(M=None):
    """Return (values, vectors) for real symmetric M (default [[2,-1],[-1,2]]).

    Vectors are columns paired with values; order, sign, and nonzero scale
    are free. Vectors must span each eigenspace, including repeated roots.
    """
    # YOUR CODE HERE
    if M is None:
        M = [[2,-1],[-1,2]]

    M = np.array(M, dtype = float)
    values, vectors = np.linalg.eigh(M)

    return values, vectors 

# print(eigensystem())


# P1 Q3: Prove trace(AB)=trace(BA) for compatible matrices.
# YOUR ANSWER HERE

# P1 Q4: Show the given C is invertible, and compute its inverse step by step.
# YOUR ANSWER HERE
def matrix_inverse(C=None):
    """Return inverse of nonsingular square C; default C=[[4,7],[2,6]]."""
    # YOUR CODE HERE
    if C is None:
        C = [[4,7],[2,6]]
    
    C = np.array(C, dtype = float)
    return np.linalg.inv(C)

# print(matrix_inverse())

# PART 2 (50 points), Q1: Normalize c*3**k/k!, derive expectation and variance.
# YOUR ANSWER HERE
def distribution_moments(rate=3.):
    """Return (c, mean, variance) for P(X=k)=c*rate**k/k!, k>=0, rate>0."""
    # YOUR CODE HERE
    c = math.exp(-rate)
    mean = float(rate)
    variance = float(rate)

    return (c, mean, variance)

# print(distribution_moments())


# P2 Q2: Show reasoning for intersection, conditional, and union probabilities.
# YOUR ANSWER HERE
def event_probabilities(p_a=.5, p_b=.3, p_a_given_b=.6):
    """Return (P(A intersect B), P(B|A), P(A union B)).

    Inputs describe a valid distribution with p_a,p_b>0.
    """
    # YOUR CODE HERE
    p_a_int_b = p_a_given_b * p_b
    p_b_given_a = p_a_int_b / p_a
    p_a_union_b = p_a + p_b - p_a_int_b

    return (p_a_int_b, p_b_given_a, p_a_union_b)

# print(event_probabilities())


# P2 Q3: Derive MLE/MAP for H,H,T,H,T, including the prior and maximization.
# YOUR ANSWER HERE
def coin_estimates(heads=3, tosses=5):
    """Return (MLE, MAP) using prior density 2*theta on [0,1].

    0<=heads<=tosses, positive integer tosses. Include boundary maxima.
    """
    # YOUR CODE HERE
    raise NotImplementedError()

# print(coin_estimates())


# P2 Q4: Derive both Gaussian MLEs for [1,3,5,7] step by step.
# YOUR ANSWER HERE
def gaussian_mle(data=None):
    """Return (mean MLE, variance MLE); default data=[1,3,5,7].

    Other inputs are finite 1D samples of length>=2 with nonzero variance.
    """
    # YOUR CODE HERE
    if data is None:
        data = [1,3,5,7]
    
    n = len(data)
    meanMLE = sum(data) / n
    mean_squared = sum(x ** 2 for x in data) / n 
    varianceMLE = mean_squared - (meanMLE ** 2)

    return (meanMLE, varianceMLE)

# print(gaussian_mle())


# P2 Q5: Derive the Gaussian posterior mode for the given observation/prior.
# YOUR ANSWER HERE
def gaussian_map(x=5., observation_variance=4., prior_mean=0., prior_variance=1.):
    """Return posterior mode for one observation and a Gaussian mean prior.

    Both variances are positive; inputs are variances, not standard deviations.
    """
    # YOUR CODE HERE
    return (prior_variance * x + observation_variance * prior_mean) / (observation_variance + prior_variance)

# print(gaussian_map())


# P2 Q6: Prove Var(X)=E[X**2]-E[X]**2.
# YOUR ANSWER HERE

# PART 3 (30 points), Q1: Define covariance and derive Var(u.T@X)=u.T@Sigma@u.
# YOUR ANSWER HERE

# P3 Q2: Describe the Gaussian generator and observed correlation.
# YOUR ANSWER HERE
def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    """Return reproducible Gaussian samples (n_samples,2) using NumPy and seed.

    Defaults: mean=[2,-1], covariance=[[3,1.2],[1.2,1]]. Custom mean has shape
    (2,), covariance (2,2) is positive definite. Do not mutate input arrays.
    """
    random = np.random.default_rng(seed)

    if mean is None:
        mean = np.array([2.0, -1.0])
    else:
        mean = np.asarray(mean, dtype = float)
    
    if covariance is None:
        covariance = np.array([[3.0, 1.2], [1.2, 1.0]])
    else:
         covariance = np.asarray(covariance, dtype = float)

    data = random.multivariate_normal(mean, covariance, size = n_samples)
    return data, mean, covariance

def plot_scatter(n_samples=500, seed=0, mean=None, covariance=None):
    data, mean_vec, cov_mat = generate_data(n_samples, seed, mean, covariance)
    X1, X2 = data[:, 0], data[:, 1]

    plt.figure(figsize=(8, 6))
    plt.scatter(X1, X2)
    plt.title("Synthetic 2D Dataset")
    plt.xlabel("Feature $X_1$")
    plt.ylabel("Feature $X_2$")
    plt.grid(True)
    plt.show()

# plot_scatter()

# Correlation structure observed: 
# - There is a positive alignment, sloping up from left to right, positive correlation
# - Decently strong linear relationship, pretty straight line, not too much deviation
# - Seems to be higher variance toward middle of the plot, could be due to higher data concentrataion

# P3 Q3: Link centering/covariance/eigendecomposition to the derivation and explain the plot.
# YOUR ANSWER HERE
def pca(X):
    """Fit NumPy PCA without mutating finite X of shape (n,2), n>=2.

    Return dict: mean (2,), centered (n,2), covariance (2,2), eigenvalues (2,),
    components (2,2), scores (n,2). Use covariance divisor n-1, descending
    eigenvalues, orthonormal eigenvectors as columns, scores=centered@components.
    Signs/tied-eigenspace bases are free. Include rank-deficient and tied cases.
    NumPy eigh/SVD are allowed; fitted library PCA is not.
    """
    # YOUR CODE HERE
    X = np.asarray(X, dtype = float)
    mean = np.mean(X, axis = 0)
    centered = X - mean 
    n = X.shape[0]
    covariance = (centered.T @ centered) / (n - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    idx = np.argsort(eigenvalues[::-1])
    eigenvalues = eigenvalues[idx]
    components = eigenvectors[:, idx]
    scores = centered @ components
    return {
        "mean": mean,
        "centered": centered, 
        "covariance": covariance,
        "eigenvalues": eigenvalues,
        "components": components,
        "scores": scores
    }

def pca_visual():
    data, _, _ = generate_data()
    pca_results = pca(data)
    mean = pca_results["mean"]
    components = pca_results["components"]
    eigenvalues = pca_results["eigenvalues"]

    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], label="Data Points")

    colors = ["red", "orange"]
    for i in range(2):
        if components[0, i] < 0:
            components[:, i] *= -1
        vector = components[:, i] * np.sqrt(eigenvalues[i])
        plt.quiver(mean[0], mean[1], vector[0], vector[1],
            angles="xy", scale_units="xy", scale=1, color=colors[i],
            label = rf"PC{i+1} ($\lambda_{{{i+1}}}={eigenvalues[i]:.2f}$)",
        )
    
    plt.plot(mean[0], mean[1], "ko", label=f"Mean {np.round(mean, 2)}")
    plt.title("PCA")
    plt.xlabel("Feature $X_1$")
    plt.ylabel("Feature $X_2$")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()

# pca_visual()

# P3 Q4: Interpret the variance fractions and dimensionality reduction tradeoff.
# YOUR ANSWER HERE
def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    # YOUR CODE HERE
    eigenvaluess = np.asarray(eigenvalues, dtype = float)
    total_variance = np.sum(eigenvaluess)
    return eigenvaluess / total_variance

def pca_outputs():
    data, _, _ = generate_data()
    pca_results = pca(data)
    eigenvalues = pca_results["eigenvalues"]
    ev_ratio = explained_variance(pca_results["eigenvalues"])

    print(f"Eigenvalues: {pca_results['eigenvalues']}")
    print(f"Explained Variance Ratio: {ev_ratio}")
    print(f"PC1 Variance Explained: {ev_ratio[0] * 100:.2f}%")
    print(f"PC2 Variance Explained: {ev_ratio[1] * 100:.2f}%")

# pca_outputs()

# P3 Q2/Q3/Q4: Provide labeled figures and discuss their meaning.
# YOUR ANSWER HERE
def make_plots(X, result):
    """Return a Matplotlib Figure (or sequence of Figures), without show/save.

    Plot samples and principal directions through their mean, and explained
    variance fractions. Label axes. Plot quality is manually reviewed.
    """
    # YOUR CODE HERE
    raise NotImplementedError()


# REFLECTION: contributions, tasks completed, and external resources used.
# YOUR ANSWER HERE

if __name__ == '__main__':
    # YOUR CODE HERE: call your functions and save/display the requested figures.
    # This block does not run when the grader imports your functions.
    # print(matrix_operations())
    # print(eigensystem())
    # print(matrix_inverse())
    # print(distribution_moments())
    # print(event_probabilities())
    # print(coin_estimates())
    # print(gaussian_mle())
    # print(gaussian_map())
    # plot_scatter()
    # pca_visual()
    # pca_outputs()
    pass
