### Boosted Tree ----

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
  add_recipe(kitchen_sink_recipe)

# fitting
tic.clearlog()
tic("Boosted Tree")

boosted_tuned <- tune_grid(
  boosted_workflow, 
  resamples = car_folds, 
  grid = boosted_grid, 
  control = control_grid(save_pred = TRUE, 
                         save_workflow = TRUE, 
                         parallel_over = "everything"))

# Pace tuning code in hear
toc(log = TRUE)

# save runtime info
time_log <- tic.log(format = FALSE)

boosted_tictoc <- tibble(model = time_log[[1]]$msg, 
                         runtime = time_log[[1]]$toc - time_log[[1]]$tic)

# Write out results & workflow

save(boosted_tuned, boosted_tictoc, file = "results/boosted_tuned.rda")