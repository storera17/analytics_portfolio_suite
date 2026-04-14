library(caret)
library(nnet)

make_neural_control <- function() {
  trainControl(
    method = "cv",
    number = 5,
    summaryFunction = twoClassSummary,
    classProbs = TRUE,
    savePredictions = "final"
  )
}

fit_neural_model <- function(data) {
  ctrl <- make_neural_control()

  caret::train(
    loan_default ~ .,
    data = data,
    method = "nnet",
    metric = "ROC",
    linout = FALSE,
    tuneGrid = expand.grid(
      size = 1:10,
      decay = seq(0.01, 0.15, by = 0.02)
    ),
    trace = FALSE,
    maxit = 300,
    trControl = ctrl
  )
}
