# {MODEL TYPE] tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(kernlab)
library(broom)

# handle common conflicts
tidymodels_prefer()

library(doMC)
registerDoMC(cores = 8)

load("data/car_split.rda")

load("data/car_folds.rda")

recipe_1 <- recipe(is_claim ~ .,
                   data = car_train) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_nzv(all_predictors()) %>% 
  step_corr(all_predictors()) %>% 
  step_normalize(all_predictors())

# Define model ----

mars_model <- mars(
  mode = "classification",
  num_terms = tune(),
  prod_degree = tune()
) %>%
  set_engine("earth")


mars_params <- extract_parameter_set_dials(mars_model) %>% 
  update(num_terms = num_terms(range = c(1, 40)))

mars_grid <- grid_regular(mars_params, levels = 5)

# workflow ----

mars_workflow <- workflow() %>% 
  add_model(mars_model) %>% 
  add_recipe(recipe_1)

mars_tune <- tune_grid(
  mars_workflow,
  resamples = car_folds,
  grid = mars_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = TRUE,
                         parallel_over = "everything")
)

final_wkflw <- mars_workflow %>% 
  finalize_workflow(select_best(mars_tune, metric = "roc_auc"))

a <- select_best(mars_tune, metric = "roc_auc")

a[3] <- 

View(show_best(mars_tune, metric = "roc_auc"))


ks_results <- fit(final_wkflw, car_train)

var_imp <- ks_results %>%
  extract_fit_parsnip() %>%
  vip::vi()

# Write out results & workflow

final_vars <- var_imp %>% 
  filter(Importance != 0, Variable != "(Intercept)") %>% 
  pull(Variable)

final_vars[3] <- "a"

final_vars


filtered_test <- filtered_test %>% 
  mutate(height = car_test$height,
         policy_tenure = car_test$policy_tenure,
         age_of_car = car_test$age_of_car,
         is_speed_alert = car_test$is_speed_alert,
         age_of_policyholder = car_test$age_of_policyholder,
         is_claim = car_test$is_claim)

filtered_train <- filtered_train %>% 
  mutate(y = factor(car_train$is_claim),
         gross_weight = car_train$gross_weight)

filtered_test <- filtered_test %>% 
  mutate(gross_weight = car_test$gross_weight)

save(filtered_train, filtered_test, file = "data/car_split_mars.rda")
