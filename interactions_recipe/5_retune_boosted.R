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
  update(mtry = mtry(c(1, 20)), 
         min_n = min_n(c(20, 60)), 
        learn_rate = learn_rate(c(.001, .1), trans = NULL)
         )

boosted_grid <- grid_regular(boosted_params, levels = 5)

# workflow ----
boosted_workflow <- workflow() %>% 
  add_model(boosted_model) %>% 
  add_recipe(interactions_recipe)

boosted_retuned <- tune_grid(
  boosted_workflow, 
  resamples = car_folds, 
  grid = boosted_grid, 
  control = control_grid(save_pred = TRUE, 
                         save_workflow = TRUE, 
                         parallel_over = "everything"))


# Write out results & workflow

save(boosted_retuned, file = "results/boosted_retuned.rda")
