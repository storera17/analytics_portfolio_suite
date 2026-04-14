library(caret)

make_xgb_control <- function() {
  trainControl(
    method = "cv",
    number = 5,
    summaryFunction = twoClassSummary,
    classProbs = TRUE,
    savePredictions = "final",
    allowParallel = TRUE
  )
}

fit_xgb_model <- function(data) {
  ctrl <- make_xgb_control()

  caret::train(
    loan_default ~ .,
    data = data,
    method = "xgbTree",
    trControl = ctrl,
    tuneLength = 5,
    metric = "ROC"
  )
}
