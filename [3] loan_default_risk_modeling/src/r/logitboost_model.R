library(caret)

make_logitboost_control <- function() {
  trainControl(
    method = "cv",
    number = 5,
    summaryFunction = twoClassSummary,
    classProbs = TRUE,
    savePredictions = "final",
    allowParallel = TRUE
  )
}

fit_logitboost_model <- function(data) {
  ctrl <- make_logitboost_control()

  caret::train(
    loan_default ~ .,
    data = data,
    method = "LogitBoost",
    trControl = ctrl,
    tuneLength = 5,
    metric = "ROC"
  )
}
