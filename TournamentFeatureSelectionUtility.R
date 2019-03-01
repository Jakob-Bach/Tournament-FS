library(data.table)

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
