# CSE 242 – Machine Learning

## Background

This assignment covers the essential mathematical and statistical foundations required for a comprehensive understanding of machine learning. It emphasizes Linear Algebra, Probability theory, Maximum Likelihood Estimation (MLE), Maximum a Posteriori (MAP) estimation, and Principal Component Analysis (PCA). A strong grasp of these mathematical tools is essential for advancing to more sophisticated machine learning topics.

Refer to the following slides provided in class for detailed explanations:

- `Lecture1_intro`
- `Lecture2_review`

## Assignment Objectives

In this assignment, you will:

- Solve detailed problems in Linear Algebra involving vectors, matrices, eigenvectors, and eigenvalues.
- Derive solutions and perform calculations related to Probability, MLE, and MAP estimations.
- Integrate mathematical understanding with practical programming in PCA.

## Grading

The assignment is worth **100 points**, graded automatically: 20 for linear algebra, 50 for probability and estimation, and 30 for PCA. Proofs, written derivations, plot presentation, and interpretation are ungraded practice; points are awarded for the corresponding numerical results and implementations. There are no pending manual grading points. The required reflection and its separate 25-point missing-reflection deduction still apply.

## Part 1: Linear Algebra (20 points)

**Q1 (5 points).** Given matrices:

$$
A = \begin{bmatrix}3 & 4 \\ 2 & 1\end{bmatrix}, \quad
B = \begin{bmatrix}1 & 2 \\ 0 & -1\end{bmatrix}
$$

Compute:

$$
AB,\quad BA,\quad (A+B)^T,\quad AB^T,\quad \operatorname{trace}(A),\quad \operatorname{trace}(AB).
$$

**Q2 (10 points).** For the following matrix:

$$
M = \begin{bmatrix}2 & -1 \\ -1 & 2\end{bmatrix}
$$

Calculate the eigenvalues and eigenvectors. Verify your solution by showing $Mv = \lambda v$.

**Q3 (Ungraded practice, 0 points).** Prove the following property of the trace operator:

$$
\operatorname{trace}(AB)=\operatorname{trace}(BA)
$$

**Q4 (5 points).** Show step-by-step that the inverse of the matrix

$$
C = \begin{bmatrix}4 & 7 \\ 2 & 6\end{bmatrix}
$$

exists, and compute its inverse.

## Part 2: Probability, MLE, and MAP Estimation (50 points)

**Q1 (10 points).** Suppose a random variable $X$ follows a discrete distribution:

$$
P(X=k)=c\cdot\frac{3^k}{k!},\quad k=0,1,2,\ldots
$$

(a) Find the normalization constant $c$.

(b) Compute the expectation $E[X]$ and variance $Var(X)$.

**Q2 (15 points).** Given two events $A$ and $B$ with $P(A)=0.5$, $P(B)=0.3$, and $P(A\mid B)=0.6$, compute the following:

(a) $P(A\cap B)$,

(b) $P(B\mid A)$,

(c) $P(A\cup B)$.

**Q3 (10 points).** Suppose you have a biased coin with unknown probability of heads $\theta=P(H)$. You flip this coin 5 times and observe the sequence: H, H, T, H, T.

(a) Derive the Maximum Likelihood Estimate (MLE) for $\theta$, showing each step from the likelihood function.

(b) Compute the Maximum a Posteriori (MAP) estimate for $\theta$, assuming a prior distribution:

$$
P(\theta)=\begin{cases}2\theta, & 0\leq\theta\leq 1 \\ 0, & \text{otherwise}\end{cases}
$$

**Q4 (7.5 points).** Consider a Gaussian distribution with parameters $\mu$ (mean) and $\sigma^2$ (variance). Given the dataset $X=\{1,3,5,7\}$, derive the Maximum Likelihood Estimates (MLE) for both $\mu$ and $\sigma^2$. Show each mathematical step clearly.

**Q5 (7.5 points).** You observe samples drawn from a Gaussian distribution with unknown mean $\mu$ and known variance $\sigma^2=4$. Given a Gaussian prior for the mean $\mu$:

$$
\mu\sim\mathcal{N}(0,1)
$$

derive the MAP estimate of $\mu$ after observing a single data point $x=5$.

**Q6 (Ungraded practice, 0 points).** Prove the variance expansion identity:

$$
Var(X)=E(X^2)-[E(X)]^2
$$

## Part 3: Principal Component Analysis (Math and Programming) (30 points)

**Q1 (Ungraded practice, 0 points).** Given a dataset represented by a random vector $X$, derive the mathematical expression for the variance of the projection onto a unit vector $u$:

$$
Var(u^TX)=u^T\Sigma u
$$

Document each intermediate mathematical step and define the covariance matrix $\Sigma$.

**Q2 (5 points).** Generate a synthetic 2-dimensional dataset composed of two correlated Gaussian features. Use NumPy for data generation. Visualize this dataset with scatter plots and discuss the correlation structure observed.

**Q3 (20 points).** Implement PCA from scratch using NumPy. Follow these steps:

- Center your dataset (mean subtraction).
- Compute the covariance matrix.
- Find eigenvalues and eigenvectors of the covariance matrix.
- Select the principal components corresponding to the highest eigenvalues.

Document each step in your implementation, linking it to your mathematical derivation from Q1. Provide visualizations of your original data along with identified principal component directions.

**Q4 (5 points).** Compute the proportion of total variance explained by each principal component. Discuss the results and explain the significance of dimensionality reduction. Provide insights into the trade-off between explained variance and dimensionality reduction.

## Submission

Submit to the autograder using:
`python3 -m autograder.run.submit <filename>`

There are instructions on installing and using the autograder on the [course README](../README.md).


## Collaboration and AI Policy

Per the course AI policy, AI usage is allowed on this assignment as long as you submit the conversation log to Canvas. 

Good luck!
