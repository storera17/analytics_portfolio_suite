library(caret)

make_ada_control <- function() {
  trainControl(
    method = "cv",
    number = 5,
    summaryFunction = twoClassSummary,
    classProbs = TRUE,
    savePredictions = "final",
    allowParallel = TRUE
  )
}

fit_adaboost_model <- function(data) {
  ctrl <- make_ada_control()

  caret::train(
    loan_default ~ .,
    data = data,
    method = "AdaBoost.M1",
    trControl = ctrl,
    tuneLength = 5,
    metric = "ROC"
  )
}
