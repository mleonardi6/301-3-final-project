### KNN ----

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

# Define model ----

knn_model <- nearest_neighbor(mode = "classification", 
                              neighbors = tune()) %>% 
  set_engine("kknn")

# set-up tuning grid ----
knn_params <- extract_parameter_set_dials(knn_model)
knn_grid <- grid_regular(knn_params, levels = 5)

# workflow ----
knn_workflow <- workflow() %>% 
  add_model(knn_model) %>% 
  add_recipe(interactions_recipe)


knn_stack <- tune_grid(
  knn_workflow, 
  resamples = car_folds, 
  grid = knn_grid, 
  control = control_stack_grid())


# Write out results & workflow

save(knn_stack, file = "results/knn_stack.rda")