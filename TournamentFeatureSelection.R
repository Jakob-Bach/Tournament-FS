library(data.table)

source("TournamentFeatureSelectionUtility.R")

###### Create data #####
# dataset <- createDemoData(numObjects = 1000, numFeatures = 100, noiseFraction = 0.1)
set.seed(25)
dataset <- caret::twoClassSim(n = 1000, linearVars = 5, noiseVars = 95, mislabel = 0.1)
dataset <- data.table(dataset)[, `:=`(target = as.integer(Class == "Class2"), Class = NULL)]

###### Train-test split (stratified) #####
set.seed(25)
class1Idx <- dataset[, which(target == "1")]
class0Idx <- dataset[, which(target == "0")]
trainClass1Idx <- sample(class1Idx, size = round(0.8 * length(class1Idx)), replace = FALSE)
trainClass0Idx <- sample(class0Idx, size = round(0.8 * length(class0Idx)), replace = FALSE)
trainData <- dataset[sort(c(trainClass0Idx, trainClass1Idx))]
testData <- dataset[-c(trainClass0Idx, trainClass1Idx)]

##### Select features #####

# All
selectedFeatures <- setdiff(names(dataset), "target")

##### Evaluate with xgboost classifier #####

xgbTrainPredictors <- Matrix::sparse.model.matrix(~ ., data = trainData[, mget(selectedFeatures)])[, -1]
xgbTrainLabels <- trainData$target
xgbTestPredictors <- Matrix::sparse.model.matrix(~ ., data = testData[, mget(selectedFeatures)])[, -1]
xgbTestLabels <- testData$target
xgbModel <- xgboost::xgboost(data = xgboost::xgb.DMatrix(label = xgbTrainLabels,
    data = xgbTrainPredictors), nrounds = 20, verbose = 2, params = list(objective = "binary:logistic"))

prediction <- predict(xgbModel, newdata = xgbTestPredictors)
mcc(as.integer(prediction >= 0.5), xgbTestLabels)
xgboost::xgb.ggplot.importance(xgboost::xgb.importance(model = xgbModel), top_n = 10)

##### Evaluate with kNN classifier #####

numericFeatures <- selectedFeatures[sapply(selectedFeatures, function(x) dataset[, is.numeric(get(x))])]
prediction <- class::knn(train = trainData[, mget(numericFeatures)],
    test = testData[, mget(numericFeatures)], cl = trainData$target, k = 10)
mcc(as.integer(as.character(prediction)), testData$target)
