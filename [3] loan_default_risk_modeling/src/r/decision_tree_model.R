library(caret)
library(dplyr)
library(rpart)
library(pROC)
library(yardstick)

cv_tree_auc <- function(data, cp, maxdepth, minsplit, minbucket, k = 5, seed = 123) {
  set.seed(seed)
  folds <- createFolds(data$loan_default, k = k, returnTrain = FALSE)

  ctrl <- rpart.control(
    cp = cp,
    maxdepth = maxdepth,
    minsplit = minsplit,
    minbucket = minbucket,
    xval = 0
  )

  oof_prob <- rep(NA_real_, nrow(data))
  aucs <- numeric(length(folds))

  for (i in seq_along(folds)) {
    val_idx <- folds[[i]]
    train_part <- data[-val_idx, , drop = FALSE]
    val_part <- data[val_idx, , drop = FALSE]

    fit <- rpart(
      loan_default ~ .,
      data = train_part,
      method = "class",
      control = ctrl
    )

    prob <- predict(fit, val_part, type = "prob")[, "Yes"]
    oof_prob[val_idx] <- prob

    roc_obj <- pROC::roc(val_part$loan_default, prob, levels = c("No", "Yes"), direction = "<")
    aucs[i] <- as.numeric(pROC::auc(roc_obj))
  }

  tibble(
    mean_auc = mean(aucs, na.rm = TRUE),
    sd_auc = sd(aucs, na.rm = TRUE),
    prob = list(oof_prob)
  )
}
