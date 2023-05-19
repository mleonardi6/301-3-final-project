library(tidyverse)
library(tidymodels)

set.seed(111)

### Load required objects
load("data/car_split.rda")

### Fold data

car_folds <- car_train %>%  vfold_cv(v = 5, repeats = 3, strata = is_claim)

### Save folds
save(car_folds, file = "data/car_folds.rda")

### Write kitchen sink recipe
kitchen_sink_recipe <- recipe(is_claim ~ ., data = car_train) %>% 
  step_rm(policy_id)
  step_dummy(all_nominal_predictors()) %>% 
  step_nzv(all_predictors()) %>%
  step_normalize(all_predictors()) %>% 
  step_corr(all_predictors())

kitchen_sink_recipe %>% 
  prep() %>% 
  bake(new_data = NULL) %>% 
  view()

skimr::skim_without_charts(car_train)
### Save recipe
save(kitchen_sink_recipe, file = "data/kitchen_sink_recipe.rda")

### Write another recipe
