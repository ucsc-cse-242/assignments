# CSE 242 — Assignment 4: Clustering and Learning with Noisy Data

## Overview

In this assignment, you will study two central ideas from this part of the course:

1. **latent-variable models and clustering**, including EM and K-means; and
2. **learning under imperfect labels**, including kernel methods and data-driven approaches for identifying or correcting noisy labels.

Unlike earlier assignments, this assignment is intentionally more open-ended. You will choose a dataset, make modeling decisions, and justify those decisions using evidence.

The goal is not to obtain the highest possible score on a benchmark. The goal is to demonstrate that you can formulate a machine-learning question, select appropriate methods, evaluate them carefully, and explain what you learned.

---

## General Requirements

- This is an **individual assignment** unless otherwise stated by the instructor.
- You may use standard Python libraries such as `numpy`, `pandas`, `matplotlib`, `scikit-learn`, and `scipy` unless a question explicitly asks you to implement an algorithm yourself.
- Your work should be reproducible. Set random seeds where appropriate and clearly describe any preprocessing.
- Clearly acknowledge any external resources, code, datasets, or people that helped you.
- Do **not** submit a Jupyter notebook.

### What to submit

Submit a single `.zip` file containing:

1. `report.pdf` — your written answers, figures, tables, and discussion;
2. your source code as one or more `.py` files; and
3. a short `README.md` explaining how to run your code and where the dataset comes from.

You may start from the optional Python starter code provided with this assignment, but you are not required to use it.

---

# Part 1 — EM for Gaussian Mixture Models (20 points)

Consider a Gaussian mixture model

$$
p(x;\theta) = \sum_{k=1}^{K} \pi_k\,\mathcal{N}(x;\mu_k,\Sigma_k),
$$

where $\pi_k \geq 0$ and $\sum_{k=1}^{K}\pi_k=1$.

## 1.1 E-step derivation (8 points)

Derive the posterior responsibility

$$
r_{ik}=P(z_i=k\mid x_i,\theta)
$$

and show that

$$
r_{ik}=\frac{\pi_k\mathcal{N}(x_i;\mu_k,\Sigma_k)}
{\sum_{j=1}^{K}\pi_j\mathcal{N}(x_i;\mu_j,\Sigma_j)}.
$$

Your derivation should make clear where Bayes' rule and the mixture-model assumptions are used.

## 1.2 M-step derivation (12 points)

Starting from the expected complete-data log likelihood, derive the M-step updates for

- $\mu_k$, and
- $\pi_k$.

You do not need to derive the covariance update.

For each major step, briefly identify the mathematical idea being used (for example: expectation, independence of observations, derivative of a sum, Lagrange multiplier, first-order optimality condition).

---

# Part 2 — Exploring Structure with K-Means (35 points)

Choose a **real dataset of your own** for this part. The dataset should contain enough numerical information for a meaningful clustering analysis. You may use a public dataset, a dataset related to your own interests, or a dataset from another scientific domain.

You should not choose a dataset simply because it already has an obvious textbook K-means solution.

## 2.1 Dataset and question (5 points)

Describe:

- the dataset and its source;
- what each observation represents;
- the features you will use;
- any preprocessing or feature scaling;
- why clustering might reveal something interesting about this dataset; and
- one concrete question you hope the clustering analysis will help you investigate.

If the dataset contains known labels, do **not** use those labels to fit K-means. You may use them later for interpretation.

## 2.2 Implement K-means from scratch (12 points)

Implement K-means yourself. Your implementation must include, at minimum:

- initialization of centroids;
- assignment of points to the closest centroid;
- centroid updates;
- a stopping criterion; and
- a maximum number of iterations.

You may use `numpy`, but you may not call a library K-means implementation for this section.

Briefly describe how you handle edge cases such as empty clusters.

## 2.3 Choosing and interpreting $K$ (10 points)

Evaluate at least **four** different values of $K$.

Use at least one quantitative diagnostic, such as:

- within-cluster sum of squares;
- silhouette score;
- stability across random initializations; or
- another metric you justify.

Then choose a value (or range of values) of $K$ that you believe is most informative for your dataset.

Your choice does not need to correspond to the numerically best metric value if you can justify why another solution is more useful or interpretable.

## 2.4 Visualization and analysis (8 points)

Create at least one visualization that helps explain the clusters. If your data are higher-dimensional, you may use PCA or another dimensionality-reduction method **for visualization only**.

Discuss:

- what distinguishes the clusters;
- whether the clustering supports your original question;
- whether some points appear ambiguous or poorly represented by hard cluster assignments; and
- at least one limitation of applying K-means to this dataset.

---

# Part 3 — Classification with Noisy Labels (45 points)

Choose a **supervised classification dataset of your own**. This may be the same dataset as Part 2 if it contains an appropriate target label, or a different dataset.

Your task is to investigate how label noise affects a classifier and whether a data-driven strategy can reduce the damage.

## 3.1 Baseline problem and evaluation design (8 points)

Describe:

- the prediction task;
- the target classes;
- the features;
- your train/validation/test split; and
- the metric(s) you will use.

Train a baseline classifier on the **clean training data**. Use an SVM with either a linear or RBF kernel unless you have a strong reason to choose another kernel.

Tune hyperparameters using only the training/validation data. Do not tune on the test set.

## 3.2 Introduce label noise (8 points)

Create at least **two noisy-label settings**. At least one setting should use asymmetric or class-dependent noise rather than flipping every label with the same probability.

Explain why your noise model is reasonable as an experimental stress test.

For multiclass datasets, define clearly how corrupted labels are selected.

Train the same classifier on the noisy training data and compare performance with the clean-data baseline.

Because label corruption is random, repeat each noisy experiment with multiple random seeds and report a mean and a measure of variability.

## 3.3 Design a label-repair or noise-robustness strategy (15 points)

Design and evaluate **one method** intended to identify, correct, down-weight, or otherwise reduce the effect of noisy labels.

Your method must use one of the ideas below:

- **nearest neighbors**;
- **clustering**, such as K-means or a Gaussian mixture model; or
- a combination of the two.

You have substantial freedom in how you formulate the method. For example, you might:

- relabel examples whose labels strongly disagree with their neighbors;
- remove examples with low neighborhood agreement;
- identify clusters with internally inconsistent labels;
- use soft cluster membership to estimate label confidence; or
- combine local and global information.

The important part is that you explain your reasoning and make the method precise enough to reproduce.

## 3.4 Controlled comparison (8 points)

Compare at least these three conditions:

1. classifier trained on clean training labels;
2. classifier trained on noisy training labels; and
3. classifier trained after applying your repair/robustness strategy.

Use the **same clean test set** for all conditions.

Present your results in a clear table or figure and discuss whether your method helps consistently or only in certain noise regimes.

## 3.5 Reflection (6 points)

In approximately 1–2 paragraphs, answer:

- When did your strategy help?
- When did it fail or make things worse?
- What assumptions does your approach make about the geometry of the data or the relationship between features and labels?
- If this were a real-world dataset with unknown label errors, what evidence would you need before automatically changing labels?

---

# Grading Summary

| Component | Points |
|---|---:|
| Part 1: EM derivation | 20 |
| Part 2: K-means investigation | 35 |
| Part 3: Noisy-label investigation | 45 |
| **Total** | **100** |

