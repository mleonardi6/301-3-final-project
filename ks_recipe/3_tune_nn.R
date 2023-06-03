### Neural Network ----

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
load("data/kitchen_sink_recipe.rda")

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
  add_recipe(kitchen_sink_recipe)

# fitting
tic.clearlog()
tic("Neural Network")

nn_tuned <- tune_grid(
  nn_workflow, 
  resamples = car_folds, 
  grid = nn_grid, 
  control = control_grid(save_pred = TRUE, 
                         save_workflow = TRUE, 
                         parallel_over = "everything"))


# Pace tuning code in here
toc(log = TRUE)

# save runtime info
time_log <- tic.log(format = FALSE)

nn_tictoc <- tibble(model = time_log[[1]]$msg, 
                    runtime = time_log[[1]]$toc - time_log[[1]]$tic)

# Write out results & workflow

save(nn_tuned, nn_tictoc, file = "results/nn_tuned.rda")
