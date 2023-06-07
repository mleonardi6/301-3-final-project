### Neural Network ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(doMC)
library(stacks)

# handle common conflicts
tidymodels_prefer()

# load required objects ----
registerDoMC(cores = 8)
load("../data/car_folds.rda")
load("../data/car_split.rda")
load("../data/interactions_recipe.rda")

## define model
nn_model <- mlp(
  mode = "classification",
  hidden_units = tune(),
  penalty = tune()
) %>%
  set_engine("nnet")

# set-up tuning grid ----
nn_params <- extract_parameter_set_dials(nn_model)

nn_grid <- grid_regular(nn_params, levels = 5)

# workflow ----
nn_workflow <- workflow() %>% 
  add_model(nn_model) %>% 
  add_recipe(interactions_recipe)


nn_stack <- tune_grid(
  nn_workflow, 
  resamples = car_folds, 
  grid = nn_grid, 
  control = control_stack_grid())


# Write out results & workflow

save(nn_stack, file = "results/nn_stack.rda")
