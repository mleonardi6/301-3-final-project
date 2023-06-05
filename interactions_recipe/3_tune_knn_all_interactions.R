### KNN ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(doMC)
library(tictoc)

# handle common conflicts
tidymodels_prefer()

# load required objects ----
registerDoMC(cores = 4)
load("../data/car_folds.rda")
load("../data/car_split.rda")
load("../data/interactions_recipe2.rda")

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
  add_recipe(interactions_recipe2)

# fitting
tic.clearlog()
tic("KNN")

knn_tuned <- tune_grid(
  knn_workflow, 
  resamples = car_folds, 
  grid = knn_grid, 
  control = control_grid(save_pred = TRUE, 
                         save_workflow = TRUE, 
                         parallel_over = "everything"))

# Pace tuning code in hear
toc(log = TRUE)

# save runtime info
time_log <- tic.log(format = FALSE)

knn_tictoc <- tibble(model = time_log[[1]]$msg, 
                     runtime = time_log[[1]]$toc - time_log[[1]]$tic)

# Write out results & workflow

save(knn_tuned, knn_tictoc, file = "results/knn_tuned_all_interactions.rda")