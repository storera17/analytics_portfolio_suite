library(caret)
library(dplyr)
library(purrr)
library(randomForest)
library(recipes)
library(pROC)
library(tibble)

default_target <- "target"
default_positive <- "Yes"
default_negative <- "No"

normalize_target_labels <- function(data,
                                    target_col = default_target,
                                    positive_label = default_positive,
                                    negative_label = default_negative) {
  data[[target_col]] <- factor(data[[target_col]], levels = c(negative_label, positive_label))
  data
}

make_balanced_subset <- function(data,
                                 target_col = default_target,
                                 positive_label = default_positive,
                                 negative_label = default_negative,
                                 seed = 123) {
  set.seed(seed)

  positive_rows <- data %>% filter(.data[[target_col]] == positive_label)
  negative_rows <- data %>% filter(.data[[target_col]] == negative_label)

  minority_n <- min(nrow(positive_rows), nrow(negative_rows))

  positive_draw <- positive_rows %>% sample_n(minority_n)
  negative_draw <- negative_rows %>% sample_n(minority_n)

  bind_rows(positive_draw, negative_draw) %>% slice_sample(prop = 1)
}

make_recipe_standard <- function(data, target_col = default_target) {
  recipe(as.formula(paste(target_col, "~ .")), data = data) %>%
    step_impute_median(all_numeric_predictors()) %>%
    step_impute_mode(all_nominal_predictors()) %>%
    step_dummy(all_nominal_predictors()) %>%
    step_normalize(all_numeric_predictors())
}

prof_confusion <- function(truth, pred, positive = default_positive) {
  cm <- caret::confusionMatrix(data = pred, reference = truth, positive = positive)
  print(cm)
  invisible(cm)
}

score_threshold <- function(truth,
                            prob_yes,
                            threshold,
                            positive_label = default_positive,
                            negative_label = default_negative) {
  pred <- factor(
    ifelse(prob_yes >= threshold, positive_label, negative_label),
    levels = c(negative_label, positive_label)
  )

  roc_obj <- pROC::roc(truth, prob_yes, levels = c(negative_label, positive_label), direction = "<")

  tibble(
    threshold = threshold,
    f1 = MLmetrics::F1_Score(y_pred = pred, y_true = truth, positive = positive_label),
    precision = posPredValue(pred, truth, positive = positive_label),
    recall = sensitivity(pred, truth, positive = positive_label),
    accuracy = mean(pred == truth),
    auc = as.numeric(pROC::auc(roc_obj))
  )
}

find_best_f1_threshold <- function(truth,
                                   prob_yes,
                                   positive_label = default_positive,
                                   negative_label = default_negative) {
  curve <- bind_rows(lapply(seq(0.01, 0.99, by = 0.01), function(thr) {
    score_threshold(truth, prob_yes, thr, positive_label, negative_label)
  }))

  best <- curve %>% slice_max(f1, n = 1, with_ties = FALSE)

  list(
    best_threshold = best$threshold[[1]],
    curve = curve,
    best_row = best
  )
}

make_forest_control <- function(folds = 5) {
  trainControl(
    method = "cv",
    number = folds,
    summaryFunction = twoClassSummary,
    classProbs = TRUE,
    savePredictions = "final"
  )
}

make_forest_grid <- function() {
  expand.grid(mtry = c(4, 6, 8, 10))
}

fit_generic_forest <- function(data,
                               target_col = default_target,
                               folds = 5,
                               n_trees = 300,
                               min_node_size = 5,
                               metric_name = "ROC",
                               seed = 123) {
  set.seed(seed)

  ctrl <- make_forest_control(folds = folds)
  grid <- make_forest_grid()

  formula_obj <- as.formula(paste(target_col, "~ ."))

  caret::train(
    formula_obj,
    data = data,
    method = "rf",
    trControl = ctrl,
    tuneGrid = grid,
    ntree = n_trees,
    nodesize = min_node_size,
    importance = TRUE,
    metric = metric_name
  )
}

extract_cv_probabilities <- function(fitted_model,
                                     positive_label = default_positive) {
  fitted_model$pred %>%
    arrange(rowIndex) %>%
    pull(.data[[positive_label]])
}

extract_cv_truth <- function(fitted_model, target_col = default_target) {
  fitted_model$pred %>%
    arrange(rowIndex) %>%
    pull(obs)
}

evaluate_on_holdout <- function(fitted_model,
                                holdout_data,
                                threshold,
                                target_col = default_target,
                                positive_label = default_positive,
                                negative_label = default_negative) {
  truth <- holdout_data[[target_col]]
  prob_yes <- predict(fitted_model, newdata = holdout_data, type = "prob")[, positive_label]

  pred <- factor(
    ifelse(prob_yes >= threshold, positive_label, negative_label),
    levels = c(negative_label, positive_label)
  )

  cm <- caret::confusionMatrix(pred, truth, positive = positive_label)
  roc_obj <- pROC::roc(truth, prob_yes, levels = c(negative_label, positive_label), direction = "<")

  tibble(
    threshold = threshold,
    accuracy = unname(cm$overall["Accuracy"]),
    precision = unname(cm$byClass["Precision"]),
    recall = unname(cm$byClass["Recall"]),
    specificity = unname(cm$byClass["Specificity"]),
    f1 = 2 * ((cm$byClass["Precision"] * cm$byClass["Recall"]) /
                (cm$byClass["Precision"] + cm$byClass["Recall"])),
    auc = as.numeric(pROC::auc(roc_obj))
  )
}

extract_feature_importance <- function(fitted_model) {
  raw_imp <- varImp(fitted_model)$importance %>%
    tibble::rownames_to_column("feature")

  names(raw_imp)[2] <- "importance"

  raw_imp %>%
    arrange(desc(importance))
}

select_top_features <- function(importance_table,
                                min_importance = 0,
                                top_n = NULL) {
  selected <- importance_table %>%
    filter(importance > min_importance)

  if (!is.null(top_n)) {
    selected <- selected %>% slice_head(n = top_n)
  }

  selected$feature
}

fit_generic_forest_workflow <- function(train_data,
                                        holdout_data = NULL,
                                        target_col = default_target,
                                        positive_label = default_positive,
                                        negative_label = default_negative,
                                        folds = 5,
                                        n_trees = 300,
                                        min_node_size = 5,
                                        seed = 123) {
  model_fit <- fit_generic_forest(
    data = train_data,
    target_col = target_col,
    folds = folds,
    n_trees = n_trees,
    min_node_size = min_node_size,
    metric_name = "ROC",
    seed = seed
  )

  cv_truth <- extract_cv_truth(model_fit, target_col = target_col)
  cv_prob <- extract_cv_probabilities(model_fit, positive_label = positive_label)
  threshold_result <- find_best_f1_threshold(
    truth = cv_truth,
    prob_yes = cv_prob,
    positive_label = positive_label,
    negative_label = negative_label
  )

  output <- list(
    fitted_model = model_fit,
    best_threshold = threshold_result$best_threshold,
    cv_summary = threshold_result$best_row,
    feature_importance = extract_feature_importance(model_fit)
  )

  if (!is.null(holdout_data)) {
    output$holdout_summary <- evaluate_on_holdout(
      fitted_model = model_fit,
      holdout_data = holdout_data,
      threshold = threshold_result$best_threshold,
      target_col = target_col,
      positive_label = positive_label,
      negative_label = negative_label
    )
  }

  output
}
