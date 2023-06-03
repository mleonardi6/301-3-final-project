### SVM Radial ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(doMC)
library(tictoc)

# handle common conflicts
tidymodels_prefer()

# load required objects ----
registerDoMC(cores = 4)
load("data/car_folds.rda")
load("data/car_split.rda")
load("data/interactions_recipe.rda")

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
  add_recipe(kitchen_sink_recipe)

# fitting
tic.clearlog()
tic("SVM Radial")

svm_radial_tuned <- tune_grid(
  svm_radial_workflow, 
  resamples =  car_folds, 
  grid = svm_radial_grid, 
  control = control_grid(save_pred = TRUE, 
                         save_workflow = TRUE, 
                         parallel_over = "everything"))

# Pace tuning code in here
toc(log = TRUE)

# save runtime info
time_log <- tic.log(format = FALSE)

svm_radial_tictoc <- tibble(model = time_log[[1]]$msg, 
                            runtime = time_log[[1]]$toc - time_log[[1]]$tic)

# Write out results & workflow

save(svm_radial_tuned, svm_radial_tictoc, file = "results/svm_radial_tuned2.rda")
