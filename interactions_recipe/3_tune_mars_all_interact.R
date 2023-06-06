### MARS ---

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(doMC)
library(tictoc)

# handle common conflicts
tidymodels_prefer()

# load required objects ----
registerDoMC(cores = 8)
load("../data/car_folds.rda")
load("../data/car_split.rda")
load("../data/interactions_recipe2.rda")

## define model
mars_model <- mars(
  mode = "classification",
  num_terms = tune(),
  prod_degree = tune()
) %>%
  set_engine("earth")

# set-up tuning grid ----
mars_params <- extract_parameter_set_dials(mars_model) %>% 
  update(num_terms = num_terms(range = c(1, 40)))

mars_grid <- grid_regular(mars_params, levels = 5)

# workflow ----
mars_workflow <- workflow() %>% 
  add_model(mars_model) %>% 
  add_recipe(interactions_recipe2)

# fitting
tic.clearlog()
tic("MARS")

mars_tuned <- tune_grid(
  mars_workflow, 
  resamples = car_folds, 
  grid = mars_grid, 
  control = control_grid(save_pred = TRUE, 
                         save_workflow = TRUE, 
                         parallel_over = "everything"))


# Pace tuning code in here
toc(log = TRUE)

# save runtime info
time_log <- tic.log(format = FALSE)

mars_tictoc <- tibble(model = time_log[[1]]$msg, 
                      runtime = time_log[[1]]$toc - time_log[[1]]$tic)

# Write out results & workflow

save(mars_tuned, mars_tictoc, file = "results/mars_tuned2.rda")