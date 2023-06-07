### Boosted Tree ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(doMC)
library(stacks)

# handle common conflicts
tidymodels_prefer()

# load required objects ----
registerDoMC(cores = 4)
load("../data/car_folds.rda")
load("../data/car_split.rda")
load("../data/interactions_recipe.rda")

# Define model ----

boosted_model <- boost_tree(mode = "classification", 
                            learn_rate = tune(), 
                            min_n = tune(),
                            mtry = tune()) %>% 
  set_engine("xgboost", importance = "impurity")

# set-up tuning grid ----
boosted_params <- extract_parameter_set_dials(boosted_model) %>% 
  update(mtry = mtry(c(1, 40)))


boosted_grid <- grid_regular(boosted_params, levels = 5)

# workflow ----
boosted_workflow <- workflow() %>% 
  add_model(boosted_model) %>% 
  add_recipe(interactions_recipe)


boosted_stack <- tune_grid(
  object = boosted_workflow, 
  resamples = car_folds, 
  grid = boosted_grid, 
  control = control_stack_grid())

# Write out results & workflow

save(boosted_stack, file = "results/boosted_stack.rda")