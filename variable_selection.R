library(tidyverse)
library(tidymodels)
library(ggplot2)
library(naniar)

library(doMC)
registerDoMC(cores = 8)

load("data/car_split.rda")

load("data/car_folds.rda")


lasso_model <- logistic_reg(penalty = tune(), mixture = 1) %>% 
  set_engine("glmnet") %>% 
  set_mode("classification")

lasso_params <- extract_parameter_set_dials(lasso_model) %>% 
  update(penalty = penalty(range = c(0.1, 0.4), trans = NULL))

lasso_grid <- grid_regular(lasso_params, levels = 5)


recipe_1 <- recipe(is_claim ~ .,
                   data = car_train) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_nzv(all_predictors()) %>% 
  step_corr(all_predictors()) %>% 
  step_normalize(all_predictors())


lasso_wkflw <- workflow() %>% 
  add_model(lasso_model) %>% 
  add_recipe(recipe_1)

lasso_tune <- tune_grid(
  lasso_wkflw, 
  resamples = car_folds, 
  grid = lasso_grid,
  control = control_grid(parallel_over = "everything"), 
)

final_wkflw <- lasso_wkflw %>% 
  finalize_workflow(select_best(lasso_tune, metric = "roc_auc"))

ks_results <- fit(final_wkflw, car_train)

save(ks_results, file = "results/lasso_vars2.rda")


load("results/lasso_vars2.rda")

lasso_vars <- tidy(ks_results)

lasso_vars

save(lasso_vars, file = "rdas/lasso_vars2.rda")

final_vars <- lasso_vars %>% 
  filter(estimate != 0, term != "(Intercept)") %>% 
  pull(term)

final_vars

filtered_train <- train %>% 
  select(all_of(final_vars))

filtered_train <- filtered_train %>% 
  mutate(y = factor(train$y))

save(filtered_train, file = "data/filtered_train.rda")



