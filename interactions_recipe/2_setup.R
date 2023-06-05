library(tidyverse)
library(tidymodels)

set.seed(111)

### Load required objects
load("data/car_split.rda")

load("data/car_folds.rda")

### Write interactions recipe
interactions_recipe <- recipe(is_claim ~ ., data = car_train) %>% 
  step_rm(policy_id) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_interact(~ age_of_car:age_of_policyholder + displacement: length
                + turning_radius:width + height:gross_weight + airbags:age_of_car 
                + policy_tenure:age_of_policyholder + policy_tenure:age_of_car) %>% 
  step_nzv(all_predictors()) %>%
  step_normalize(all_predictors()) %>% 
  step_corr(all_predictors())

interactions_recipe %>% 
  prep() %>% 
  bake(new_data = NULL) %>% 
  view()

skimr::skim_without_charts(car_train)
### Save recipe
save(interactions_recipe, file = "data/interactions_recipe.rda")

### Write another recipe

interactions_recipe2 <- recipe(is_claim ~ ., data = car_train) %>% 
  step_rm(policy_id) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_interact(~all_numeric_predictors()^2) %>% 
  step_nzv(all_predictors()) %>%
  step_normalize(all_predictors()) %>% 
  step_corr(all_predictors())

interactions_recipe2 %>% 
  prep() %>% 
  bake(new_data = NULL) %>% 
  view()

### Save recipe
save(interactions_recipe2, file = "data/interactions_recipe2.rda")
