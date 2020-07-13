# Tournament Feature Selection

A hierarchical, iterative approach for feature selection.
(Also, a discontinued research idea.)
It evaluates the usefulness of features randomly partitioned in small groups of size m (m is an input parameter).
For each of these groups, we determine feature importance and select a subset of features (see below how we do this).
The selected features are joined, split again in groups of size m, their importance is evaluated etc.
This is inspired by sports tournaments (hierarchical elimination) and crowdsourcing (make decisions based on small tasks).
The code is an early prototype at best.
Also, we haven't versioned our R packages / the environment, so you need to install stuff manually.
 
`TournamentFeatureSelection.R ` provides:
 
- a train-test split on a demo dataset
- calls to Tournament FS and genetic-algorithm FS
- calls to three classifiers: kNN, SVM and `xgboost`
 
The script is intended for interactive use, not for running it completely.
 
`TournamentFeatureSelectionUtility.R ` provides several functions:

- the generator for a demo (binary classification) dataset where it is clear which features are useful and which are not (target variable is Boolean combination of decision-tree-like splits on some features), with a certain amount of noise (target variable flipped)
- classification performance measures (MCC and a penalized version considering the number of selected features)
- variants of tournament FS:
  - *classification-score-based*: For each feature group of size m, run a classifier (implemented: kNN, logistic regression, Naive Bayes, SVM, `xgboost`) for all feature subsets of size k (input parameter). Aggregate classification performance:
    - *local max*: Per group, select the feature set of size k with the highest classification performance.
    - *local mean*: For each feature, take the average classification performance over all feature subsets of size k the feature is involved in. Per group, select the k features with the highest average classification performance.
    - *local median*: As before, just using the median instead of the mean.
    - *global mean* (median, max would also be possible): For each feature, take the average classification performance over all feature subsets of size k the feature is involved in. Instead of ranking this classification performance locally per group, create a global ranking over all groups and select the top k \* m features.
  - *penalized version of classification-score-based*: Instead of just trying feature sets of size k per group, try all feature subsets of sizes 1..m. However, classification performance is penalized linearly with the number of features. The selection of the best feature subset per group works as in the unpenalized classification scoring (i.e., there are several options).
  - *importance-score-based*: For each group of size m, run a classifier (here: `xgboost`) with an internal importance measure once (without running it for smaller subsets). Aggregate this:
    - *local `xgboost`-based*: From each group, select k features with the highest importance.
    - *global `xgboost`-based*: Instead of doing an importance ranking locally per group, create a global ranking over all groups and select the top k \* m features.
  - *model-based* with `xgboost`: For each group of size m, train a tree-based `xgboost` model and select all features which occur in any of the trees. Thus, the number of features selected per group is variable.
