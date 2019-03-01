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

# Genetic
gaFunctions <- caret::caretGA
gaFunctions$fitness_extern <- caretMCC # else some default summary computed from which we don't get our metric
gaFunctions$fitness_intern <- caretPenalizedMCC # else only access to metric used for model
computingCluster <- parallel::makeCluster(parallel::detectCores())
doParallel::registerDoParallel(computingCluster)
set.seed(25)
gaResult <- caret::gafs(
  x = trainData[, -"target"], y = factor(trainData$target, levels = c("0", "1")),
  iters = 10, popSize = 50, pcrossover = 0.8, pmutation = 0.1, elite = 0, # default GA parameters
  differences = FALSE, # does not seem to work
  gafsControl = caret::gafsControl(functions = gaFunctions, verbose = TRUE,
      method = "boot", number = 1, seeds = c(25, 25), # although seeds used, not deterministic if parallelization enabled
      genParallel = TRUE, allowParallel = FALSE, # outer resampling (repeat whole GA) not parallel, but fitness evaluation
      metric = c(internal = "PenMCC", external = "MCC")),
  # following parameters are passed to caret::train
  # method = "xgbTree", metric = "MCC", nthread = 1,
  # tuneGrid = data.frame(nrounds = 20, max_depth = 6, eta = 0.3, gamma = 0,
  #     colsample_bytree = 1, min_child_weight = 1, subsample = 1), # need to specify this or else parameter tuning
  method = "knn", metric = "MCC", tuneGrid = data.frame(k = 10),
  trControl = caret::trainControl(method = "boot", number = 1, seeds = list(25, 25),# method = "none" does not work
      returnData = FALSE, summaryFunction = caretMCC, allowParallel = FALSE)) # inner resampling (model parameter tuning) not parallel
parallel::stopCluster(computingCluster)
print(gaResult) # final model trained on whole data (if not holdout used), after using outer resampling to estimate GA performance
# "optvariables" selected from final GA, but iteration with best external
# performance in preceding resampling (and this being not overall best external,
# but within iteration simply external belonging to highest internal)
selectedFeatures <- gaResult$optVariables
# "ga$final" selected from final GA, iteration with best internal performance
selectedFeatures <- gaResult$ga$final
plot(gaResult) # results from outer resampling

# Tournament
selectedFeatures <- tournamentFS(dataTable = trainData, k = 5, m = 10)

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

##### Evaluate with SVM classifier #####

svmModel <- kernlab::ksvm(x = target ~ ., type = "C-svc",
    data = trainData[,  mget(c(selectedFeatures, "target"))])
prediction <- kernlab::predict(svmModel, newdata = testData, type = "response")
mcc(prediction, testData$target)
