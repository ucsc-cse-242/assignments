# CSE 242 — Assignment 5: Machine Learning Mini-Project

## Overview

Assignment 5 is a small, open-ended machine-learning project. You will choose a dataset and investigate a question that interests you using methods from the second half of the course.

The project has two required technical components:

1. **Gaussian mixture models / EM**, which you will implement and use to study latent structure in your data; and
2. **neural networks**, which you will use for a supervised prediction problem and analyze experimentally.

The emphasis is on the complete ML process: defining a question, understanding the data, implementing core algorithms, making and justifying design choices, evaluating models appropriately, and communicating what you learned.

You are **not** expected to invent a new machine-learning algorithm. A strong project asks a clear question and answers it through careful experiments.

---

## Project Rules

### Dataset

Choose a dataset of your own. It should:

- be large enough to support meaningful train/validation/test evaluation;
- contain multiple useful input features;
- support a supervised prediction task; and
- be suitable for an unsupervised analysis of its feature space.

Public datasets are encouraged. You may also use a dataset connected to your research or interests, provided you are allowed to share enough information for the project to be graded.

Avoid datasets used extensively as worked examples in lecture unless your project asks a substantially different question.

### Tools

You may use standard Python libraries for data handling, visualization, evaluation, and comparison models.

However:

- the required EM algorithm must be implemented by you;
- the required neural-network forward pass and backpropagation must be implemented by you using basic array operations; and
- you may additionally use `scikit-learn`, PyTorch, TensorFlow, or another library for comparison experiments.

### Submission

Do **not** submit a Jupyter notebook.

Submit a single `.zip` file containing:

1. `report.pdf` — approximately 6–10 pages, excluding references and optional appendix;
2. your source code as `.py` files;
3. `README.md` with instructions for reproducing the main results; and
4. `requirements.txt` or an equivalent description of dependencies if needed.

If the dataset is small and redistribution is allowed, you may include it. Otherwise, provide a stable link and download/preparation instructions.

You may use the optional Python starter code, but it is not required.

---

# Part 1 — Project Proposal and Data Understanding (15 points)

## 1.1 Research question (5 points)

State one primary question your project will investigate.

A good question should be more specific than “Which model performs best?” Examples include:

- Do naturally occurring clusters correspond to meaningful subpopulations in the data?
- Does a neural network benefit from the same feature representation that produces coherent clusters?
- Which kinds of examples are consistently difficult for the supervised model, and do they occupy unusual regions of the feature space?
- How sensitive are conclusions to model capacity, regularization, or the amount of training data?

You may formulate a different question appropriate to your dataset.

## 1.2 Dataset description (5 points)

Describe:

- the source of the dataset;
- what one observation represents;
- input features and target variable;
- dataset size;
- missing values, class imbalance, or other relevant properties; and
- any ethical, privacy, measurement, or sampling concerns that matter for interpretation.

## 1.3 Experimental plan (5 points)

Before presenting results, describe your evaluation plan:

- train/validation/test split or cross-validation procedure;
- preprocessing;
- evaluation metric(s);
- what information is allowed to influence model selection; and
- which comparisons you consider necessary to answer your question.

---

# Part 2 — Latent Structure with EM and Gaussian Mixture Models (25 points)

## 2.1 Implement EM for a Gaussian mixture model (12 points)

Implement a Gaussian mixture model using the EM algorithm.

Your implementation should include:

- initialization of mixture weights, means, and covariance parameters;
- an E-step that computes posterior responsibilities;
- an M-step that updates the parameters;
- a convergence criterion or iteration limit; and
- numerical safeguards where appropriate.

You may choose full, diagonal, or spherical covariance matrices. State your choice and justify it in relation to the dataset and computational cost.

Do not call a library GMM implementation for this subsection.

## 2.2 Model selection and diagnostics (6 points)

Fit your GMM for at least **three** choices of the number of components.

Use at least one principled diagnostic to compare them, such as:

- held-out log likelihood;
- BIC;
- AIC;
- stability across initialization; or
- another measure you justify.

Discuss whether a single “best” number of components is clearly supported.

## 2.3 Interpret the latent structure (7 points)

Use visualizations and/or summary statistics to explain what the mixture components capture.

If your dataset has known labels, you may compare those labels with the inferred components **after fitting**, but the labels should not be used to train the GMM.

Discuss at least one difference between the interpretation provided by a GMM and the hard-clustering view of K-means.

Connect this analysis back to your project question.

---

# Part 3 — Neural Network from Scratch (25 points)

For this part, construct a small feed-forward neural network appropriate for your supervised task.

The network does not need to be large. The purpose is to demonstrate that you understand the mechanics of forward propagation, loss computation, backpropagation, and optimization.

## 3.1 Architecture and forward pass (5 points)

Define your network architecture and implement its forward pass using basic array operations.

At minimum, use:

- one hidden layer;
- a nonlinear activation function; and
- an output/loss appropriate for your task.

Explain the dimensions of each weight matrix and the role of each activation.

## 3.2 Backpropagation (8 points)

Derive the gradient equations needed to train your network and implement them yourself.

Your report should show the key chain-rule steps for at least one hidden-layer weight and one output-layer weight. You do not need to write out every scalar derivative if the matrix form is clear.

## 3.3 Optimization and training behavior (6 points)

Train the network using gradient descent, stochastic gradient descent, or mini-batch gradient descent.

Plot training behavior over time. Depending on your task, this could include training and validation loss, accuracy, or another relevant metric.

Explain your choices for learning rate, number of epochs, batch size, initialization, and stopping criterion.

## 3.4 Sanity checks (6 points)

Demonstrate that your implementation is behaving sensibly. Include at least two checks, such as:

- verifying gradient values numerically on a small example;
- showing that the loss decreases on a small subset of the data;
- deliberately overfitting a tiny training set;
- comparing predictions against a simple known case; or
- comparing your implementation with a library implementation using the same small architecture.

---

# Part 4 — Project Experiments (25 points)

This is the most open-ended part of the assignment.

Design a set of experiments that use your neural network to answer the primary project question from Part 1.

## Minimum required comparisons

Your experiments must include:

1. **one simple baseline** appropriate to the task;
2. your **from-scratch neural network**;
3. at least **two meaningful neural-network variants**; and
4. at least one analysis connecting the supervised results to the latent structure explored in Part 2.

A variant should change something conceptually meaningful, for example:

- hidden-layer width or depth;
- activation function;
- regularization;
- amount of training data;
- feature set;
- class weighting;
- optimization settings; or
- a representation derived from the GMM analysis.

Changing only a random seed does not count as a model variant.

## 4.1 Results (10 points)

Present the main quantitative results clearly in one or more tables or figures.

Use multiple random seeds when randomness materially affects the result. Report variability, not only the best run.

## 4.2 Error and subgroup analysis (8 points)

Go beyond aggregate performance.

Investigate where the model succeeds and fails. Your analysis could consider:

- particular classes;
- GMM components;
- confidence levels;
- feature ranges;
- rare examples;
- misclassified examples; or
- another meaningful subgroup.

Include at least one visualization that helps explain model behavior.

## 4.3 Interpretation (7 points)

Explain what the experiments tell you about your original question.

Distinguish between:

- conclusions directly supported by your experiments;
- plausible explanations; and
- questions that remain unresolved.

---

# Part 5 — Final Reflection and Reproducibility (10 points)

## 5.1 Conclusions and limitations (6 points)

Summarize your main findings in approximately 2–4 paragraphs.

Discuss at least two limitations. These might involve the dataset, experimental design, model assumptions, compute budget, evaluation metric, or generalizability.

Describe one next experiment you would run if you had more time.

## 5.2 Reproducibility and code quality (4 points)

Your submission should make it possible for another student to reproduce your main results without reconstructing your workflow from scratch.

Your `README.md` should include:

- dataset access instructions;
- environment/dependency information;
- commands for running the main experiments; and
- expected outputs or filenames.

Use functions and meaningful variable names rather than one long script whenever practical.

---

# Grading Summary

| Component | Points |
|---|---:|
| Part 1: Proposal and data understanding | 15 |
| Part 2: EM / Gaussian mixture model | 25 |
| Part 3: Neural network from scratch | 25 |
| Part 4: Project experiments | 25 |
| Part 5: Reflection and reproducibility | 10 |
| **Total** | **100** |

