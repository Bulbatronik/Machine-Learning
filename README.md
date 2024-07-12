<h1 align="center">Machine Learning </h1>

The repository provides a set of notebooks used during the *Machine Learning* course. The notebooks can easily be run on `Colab` and cover multiple topics. The topics include the state-of-art of `machine learning (ML)` and `reinforcement learning (RL)` algorithms. In particular, the topics covered by the notebooks are the following:

- [**Introduction to Python**](Notebooks/00_An%20Introduction%20to%20Python.ipynb) - the notebook provides examples with basic data structures and their manipulations. It provides examples of creations of functions, classes, as well as the most common ML-related libraries, like [*numpy*](https://numpy.org/), [*pandas*](https://pandas.pydata.org/), [*matplotlib*](https://matplotlib.org/), [*Scikit-Learn*](https://scikit-learn.org/stable/).

- [**Regression**](Notebooks/01_Linear_Regression.ipynb) - provides a comparison between the linear regression from *Scikit-Learn* and *statsmodels* and straightforward implementation, as well as the impact of regularization (`Ridge and Lasso`). Gives an example of model evaluation, like overall significance of the model, coefficients statistical test and etc.
- [**Classification**](Notebooks/02_Classification.ipynb) - discusses teh main classification algorithms for discriminant function (`perceptron`), probabilistic discriminative (`logistic regression`) and probabilistic generative (`naive Bayes`) approaches.
- [**Bias-Variance Dilemma**](Notebooks/03-Bias-Variance%20Tradeoff.ipynb) - phenomenon of the error decomposition for different models and demonstration of how different hypothesis space and number of samples influences the final performance of the models.
- [**Model Selection**](Notebooks/04-Model-Selection.ipynb) - 
demonstration of feature selection methods using backward approach, dimensionality reduction using `principle component analysis (PCA)` and reguralization.
- [**Kernel Methods**](Notebooks/05_Kernel%20Methods.ipynb) - in particular, this notebook focuses on `Gaussian process (GP)` for regression problem and `support vector machines (SVM)` for a classification tasks.
- [**Markov Decision Processes**](Notebooks/06_MDP.ipynb) - a demonstration how various decision making problem can be modeled as markov decision problem (MDP) and how to find a solution for the problems of prediction and control using `Bellman expectation equation` and `Bellman optimality equation`, respectively.
- [**Reinforcement Learning**](Notebooks/07_RL.ipynb) - an application of reinforcement learning (RL) to problems with an unknown MDP for the problems of prediction (using Temporal Difference estimation) and control (using `SARSA` and `Q-learning`).
- [**Milti-armed Bandit**](Notebooks/08_MAB.ipynb) - focus on the exploration-exploitation trade-off in the problem of multi-armed bandit using examples of frequentist (`UCB1`) and Bayesian (`Thompson Sampling`).
