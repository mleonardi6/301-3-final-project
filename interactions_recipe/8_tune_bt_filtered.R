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
load("../data/filtered_folds.rda")
load("../data/car_split_mars.rda")
load("data/filtered_interact_recipe.rda")

# Define model ----

boosted_model <- boost_tree(mode = "classification", 
                            learn_rate = c(0.005, 0.02), 
                            min_n = tune(),
                            mtry = tune()) %>% 
  set_engine("xgboost", importance = "impurity")

# set-up tuning grid ----
boosted_params <- extract_parameter_set_dials(boosted_model) %>% 
  update(mtry = mtry(c(1, 5)))


boosted_grid <- grid_regular(boosted_params, levels = 5)

# workflow ----
boosted_workflow <- workflow() %>% 
  add_model(boosted_model) %>% 
  add_recipe(interactions_recipe)

# fitting
tic.clearlog()
tic("Boosted Tree")

boosted_tuned_filtered <- tune_grid(
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

boosted_tictoc_filtered <- tibble(model = time_log[[1]]$msg, 
                         runtime = time_log[[1]]$toc - time_log[[1]]$tic)

# Write out results & workflow

save(boosted_tuned_filtered, boosted_tictoc_filtered, file = "results/boosted_tuned3.rda")