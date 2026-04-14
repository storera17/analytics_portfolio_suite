library(caret)
library(randomForest)

make_forest_control <- function() {
  trainControl(
    method = "cv",
    number = 5,
    summaryFunction = twoClassSummary,
    classProbs = TRUE,
    savePredictions = "final"
  )
}

fit_forest_model <- function(data) {
  ctrl <- make_forest_control()

  caret::train(
    loan_default ~ .,
    data = data,
    method = "rf",
    trControl = ctrl,
    tuneGrid = expand.grid(mtry = c(4, 6, 8, 10)),
    ntree = 300,
    metric = "ROC",
    importance = TRUE
  )
}
