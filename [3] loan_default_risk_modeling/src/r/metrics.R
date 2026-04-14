library(dplyr)
library(caret)
library(pROC)
library(MLmetrics)

positive_label <- "Yes"
negative_label <- "No"

score_table <- function(truth, prob_yes, threshold) {
  pred <- factor(ifelse(prob_yes >= threshold, positive_label, negative_label),
                 levels = c(negative_label, positive_label))

  tibble(
    threshold = threshold,
    f1 = F1_Score(y_pred = pred, y_true = truth, positive = positive_label),
    precision = posPredValue(pred, truth, positive = positive_label),
    recall = sensitivity(pred, truth, positive = positive_label),
    accuracy = mean(pred == truth),
    auc = as.numeric(pROC::auc(pROC::roc(truth, prob_yes, levels = c(negative_label, positive_label), direction = "<")))
  )
}

find_best_cutoff <- function(truth, prob_yes) {
  curve <- bind_rows(lapply(seq(0.01, 0.99, by = 0.01), function(thr) {
    score_table(truth, prob_yes, thr)
  }))
  best <- curve %>% slice_max(f1, n = 1, with_ties = FALSE)
  list(best_threshold = best$threshold[[1]], curve = curve)
}

print_confusion <- function(truth, prob_yes, threshold) {
  pred <- factor(ifelse(prob_yes >= threshold, positive_label, negative_label),
                 levels = c(negative_label, positive_label))
  print(confusionMatrix(pred, truth, positive = positive_label))
}
