### Random Forest ---------------

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
load("data/kitchen_sink_recipe.rda")

# Define model ----
rf_model <- rand_forest(min_n = tune(), mtry = tune()) %>% 
  set_engine("ranger", importance = "impurity") %>% 
  set_mode("classification")

# set-up tuning grid ----
rf_params <- extract_parameter_set_dials(rf_model) %>% 
  update(mtry = mtry(c(1, 40)))


rf_grid <- grid_regular(rf_params, levels = 5)

# workflow ----
rf_workflow <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(kitchen_sink_recipe)

# fitting
tic.clearlog()
tic("Random Forest")

rf_tuned <- tune_grid(
  rf_workflow, 
  resamples = car_folds, 
  grid = rf_grid, 
  control = control_grid(save_pred = TRUE, 
                         save_workflow = TRUE, 
                         parallel_over = "everything"))

# Pace tuning code in hear
toc(log = TRUE)

# save runtime info
time_log <- tic.log(format = FALSE)

rf_tictoc <- tibble(model = time_log[[1]]$msg, 
                    runtime = time_log[[1]]$toc - time_log[[1]]$tic)

# Write out results & workflow

save(rf_tuned, rf_tictoc, file = "results/rf_tuned.rda")