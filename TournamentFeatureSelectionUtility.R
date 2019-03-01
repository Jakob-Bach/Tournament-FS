library(data.table)
library(foreach)

# Creates a numeric dataset with the target variable being a boolean function
# of some (thresholded) predictors
createDemoData <- function(numObjects = 1000, numFeatures = 100, noiseFraction = 0.1) {
  stopifnot(numFeatures >= 5)
  set.seed(25)
  result <- data.table(matrix(rnorm(numObjects * numFeatures), nrow = numObjects))
  colnames(result) <- c(paste0("Useful", 1:5),
    sprintf(paste0("Useless%0", nchar(numFeatures), "d"), 1:(numFeatures - 5))) # leading zeros
  result[, target := as.integer((Useful1 > 0 & Useful2 > 0) | (Useful3 < 0 & (Useful4 < 0 | Useful5 < 0)))]
  flipIdx <- sample(c(FALSE, TRUE),prob = c(noiseFraction, 1 -  noiseFraction),
      size = numObjects, replace = TRUE)
  result[flipIdx, target := as.integer(1 - target)] # random perturbation
  return(result)
}

# Matthews Correlation Coefficient for two binary numeric vectors (symmetric
# measure, so assignment order of params does not really matter)
mcc <- function(prediction, actual) {
  tp <- sum(prediction == 1 & actual == 1) * 1.0 # prevent integer overflow
  tn <- sum(prediction == 0 & actual == 0) * 1.0
  fp <- sum(prediction == 1 & actual == 0) * 1.0
  fn <- sum(prediction == 0 & actual == 1) * 1.0
  result <- (tp*tn - fp*fn) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
  if (is.na(result)) { # just one class in actual or prediction
    result <- 0
  }
  return(result)
}

# Function which can be used as "summaryFunction" in caret::trainControl()
caretMCC <- function(data, lev = NULL, model = NULL) {
  # no calls to other (own) functions so it can also be used with caret's parallelization
  tp <- sum(data$obs == "1" & data$pred == "1") * 1.0
  tn <- sum(data$obs == "0" & data$pred == "0") * 1.0
  fp <- sum(data$obs == "0" & data$pred == "1") * 1.0
  fn <- sum(data$obs == "1" & data$pred == "0") * 1.0
  result <- (tp*tn - fp*fn) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
  if (is.na(result)) { # just one class in actual or prediction
    result <- 0
  }
  names(result) <- "MCC"
  return(result)
}

# Features selection fitness function (for caret::gafs() and caret::safs()
# penalizing number of features besides using MCC)
# based on caret::caretGA$fitness_intern()
caretPenalizedMCC <- function(object, x, y, maximize, p) {
  perf_val <- getTrainPerf(object) # returns one-row data.frame with metrics (name prefixed with "Train")
  return(c(PenMCC = perf_val$TrainMCC - ncol(x) / p)) # zero features -> no penalty, all features -> -1
}

# Maps a vector of features (column names) to a list of column names used when
# creating an xgb.DMatrix (categorical features might result in several columns)
createXgbColMapping <- function(old, new) {
  result <- lapply(old, function(colName) grep(paste0("^", colName), new, value = TRUE))
  names(result) <- old
  stopifnot(all(sapply(result, length) > 0)) # each feature has to be mapped
  stopifnot(sum(sapply(result, length)) == length(new))
  return(result)
}

# Tries all k-feature combinations on a dataset and returns the best one according
# to MCC on a holdout set
exhaustiveFS <- function(dataTable, k) {
  featureNames <- colnames(dataTable)[colnames(dataTable) != "target"]
  if (length(featureNames) < k) {
    return(featureNames)
  }
  if (!is.integer(dataTable$target) || any(dataTable$target > 1) || any(dataTable$target < 0)) {
    stop("exhaustiveFS() is written for classification with labels {0,1}.")
  }
  # Stratified train-test split
  set.seed(25)
  class1Idx <- dataTable[, which(target == "1")]
  class0Idx <- dataTable[, which(target == "0")]
  trainClass1Idx <- sample(class1Idx, size = round(0.8 * length(class1Idx)), replace = FALSE)
  trainClass0Idx <- sample(class0Idx, size = round(0.8 * length(class0Idx)), replace = FALSE)
  trainData <- dataTable[sort(c(trainClass0Idx, trainClass1Idx))]
  testData <- dataTable[-c(trainClass0Idx, trainClass1Idx)]
  xgbTrainPredictors <- Matrix::sparse.model.matrix(~ ., data = trainData[, -"target"])[, -1]
  xgbTestPredictors <- Matrix::sparse.model.matrix(~ ., data = testData[, -"target"])[, -1]
  xgbColMapping <- createXgbColMapping(old = featureNames,
      new = colnames(xgbTrainPredictors)) # necessary because of one-hot encoding
  # Create all feature combinations of size k
  featureSubsets <- combn(featureNames, m = k, simplify = FALSE)
  # Evaluate performance of feature combinations
  classifierPerformances <- sapply(featureSubsets, function(features) {
      xgbFeatures <- unlist(xgbColMapping[features]) # select corresponding xgb featurs
      xgbModel <- xgboost::xgboost(
        data = xgboost::xgb.DMatrix(data = xgbTrainPredictors[, xgbFeatures, drop = FALSE],
                                    label = trainData$target),
        nrounds = 1, verbose = 0,
        params = list(objective = "binary:logistic", nthread = 1))
      prediction <- predict(xgbModel, newdata = xgbTestPredictors[, xgbFeatures, drop = FALSE])
      return(mcc(as.integer(prediction >= 0.5), testData$target))
  })
  # Determine best feature subset
  return(featureSubsets[[which.max(classifierPerformances)]])
}

# Searches for the best combination of k features, testing them in smaller subsets
# of size m and selecting the k best from these, using a tournament-like aggregation
# structure
tournamentFS <- function(dataTable, k, m) {
  featureNames <- colnames(dataTable)[colnames(dataTable) != "target"]
  progresBar <- txtProgressBar(max = ceiling(length(featureNames) / m) *
                                 ceiling(1 / (1 - k / m)), style = 3)
  completedSubsetCount <- 0 # count previously completed inner iterations for progress bar
  showProgress <- function(n) setTxtProgressBar(progresBar, n + completedSubsetCount)
  # regarding progress bar in foreach, see https://blog.revolutionanalytics.com/2015/10/updates-to-the-foreach-package-and-its-friends.html
  while (length(featureNames) > k) {
    set.seed(length(featureNames))
    subsetIds <- 1:ceiling(length(featureNames) / m)
    featureGroupIdx <- sample(rep(subsetIds, length.out = length(featureNames)))
    computingCluster <- parallel::makeCluster(parallel::detectCores())
    doSNOW::registerDoSNOW(computingCluster)
    selectionResults <- foreach(i = subsetIds,
        .packages = "data.table", .export = c("createXgbColMapping", "exhaustiveFS", "mcc"),
        .options.snow = list(progress = showProgress)) %dopar% {
      return(exhaustiveFS(dataTable[,  mget(c(featureNames[featureGroupIdx == i], "target"))], k = k))
    }
    parallel::stopCluster(computingCluster)
    featureNames <- unlist(selectionResults) # flatten list
    completedSubsetCount <- completedSubsetCount + length(subsetIds)
  }
  return(featureNames)
}
