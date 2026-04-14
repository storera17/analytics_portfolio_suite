library(caret)
library(glmnet)

make_logistic_control <- function() {
  trainControl(
    method = "cv",
    number = 4,
    summaryFunction = twoClassSummary,
    classProbs = TRUE,
    savePredictions = "final"
  )
}

fit_logistic_model <- function(data) {
  ctrl <- make_logistic_control()

  caret::train(
    loan_default ~ .,
    data = data,
    method = "glmnet",
    trControl = ctrl,
    tuneGrid = expand.grid(
      alpha = seq(0, 1, length = 8),
      lambda = seq(0.001, 0.1, length = 5)
    ),
    metric = "ROC",
    family = "binomial"
  )
}
