library(dplyr)
library(recipes)

target_name <- "loan_default"
positive_label <- "Yes"
negative_label <- "No"

normalize_target <- function(data, target_col = target_name) {
  data[[target_col]] <- factor(data[[target_col]], levels = c(negative_label, positive_label))
  data
}

make_balanced_subset <- function(data, target_col = target_name, seed = 123) {
  set.seed(seed)

  min_data <- data %>% filter(.data[[target_col]] == positive_label)
  maj_data <- data %>% filter(.data[[target_col]] == negative_label)
  maj_under <- maj_data %>% sample_n(nrow(min_data))

  bind_rows(min_data, maj_under) %>% slice_sample(prop = 1)
}

make_recipe_standard <- function(data, target_col = target_name) {
  recipe(as.formula(paste(target_col, "~ .")), data = data) %>%
    step_impute_median(all_numeric_predictors()) %>%
    step_impute_mode(all_nominal_predictors()) %>%
    step_dummy(all_nominal_predictors()) %>%
    step_normalize(all_numeric_predictors())
}

make_recipe_range <- function(data, target_col = target_name) {
  recipe(as.formula(paste(target_col, "~ .")), data = data) %>%
    step_impute_median(all_numeric_predictors()) %>%
    step_impute_mode(all_nominal_predictors()) %>%
    step_dummy(all_nominal_predictors()) %>%
    step_range(all_numeric_predictors())
}
