### MARS ---

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
  add_recipe(interactions_recipe)


mars_stack <- tune_grid(
  mars_workflow, 
  resamples = car_folds, 
  grid = mars_grid, 
  control = control_stack_resamples())


# Write out results & workflow

save(mars_stack, file = "results/mars_stack.rda")