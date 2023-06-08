library(tidyverse)
library(tidymodels)

set.seed(111)

### Load required objects
load("data/car_split_mars.rda")

car_folds <- filtered_train %>%  vfold_cv(v = 5, repeats = 3, strata = is_claim)

save(car_folds, file = "data/filtered_folds.rda")

### Write interactions recipe
interactions_recipe <- recipe(is_claim ~ ., data = filtered_train) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_interact(~ age_of_car:age_of_policyholder + 
                height:gross_weight + policy_tenure:age_of_policyholder + 
                  policy_tenure:age_of_car) %>% 
  step_nzv(all_predictors()) %>%
  step_normalize(all_predictors()) %>% 
  step_corr(all_predictors())

interactions_recipe %>% 
  prep() %>% 
  bake(new_data = NULL) %>% 
  view()

### Save recipe
save(interactions_recipe, file = "data/filtered_interact_recipe.rda")