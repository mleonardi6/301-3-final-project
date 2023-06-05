### SVM Radial ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(doMC)

# handle common conflicts
tidymodels_prefer()

# load required objects ----
registerDoMC(cores = 8)
load("../data/car_folds.rda")
load("../data/car_split.rda")
load("../data/interactions_recipe.rda")

## define model
svm_radial_model <- svm_rbf(
  mode = "classification", 
  cost = tune(),
  rbf_sigma = tune()
) %>%
  set_engine("kernlab")

# set-up tuning grid ----
svm_radial_params <- extract_parameter_set_dials(svm_radial_model)

svm_radial_grid <- grid_regular(svm_radial_params, levels = 5)

# workflow ----
svm_radial_workflow <- workflow() %>% 
  add_model(svm_radial_model) %>% 
  add_recipe(interactions_recipe)


svm_radial_stack <- tune_grid(
  svm_radial_workflow, 
  resamples =  car_folds, 
  grid = svm_radial_grid, 
  control = control_stack_resamples())


# Write out results & workflow

save(svm_radial_stack,  file = "results/svm_radial_stack.rda")


