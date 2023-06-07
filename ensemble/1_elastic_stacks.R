### Elastic net ----

# Libraries
library(tidyverse)
library(tidymodels)
library(doMC)
library(stacks)

### Load required objects
registerDoMC(cores = 4)
load("../data/car_folds.rda")
load("../data/car_split.rda")
load("../data/interactions_recipe.rda")

### Define model
elastic_net_model <- logistic_reg(penalty = tune(), mixture = tune()) %>% 
  set_engine("glmnet") %>% 
  set_mode("classification")


### Set up tuning grid
elastic_net_params <- extract_parameter_set_dials(elastic_net_model) 

# Define tuning grid
elastic_net_grid <- grid_regular(elastic_net_params, levels = 5)

### Workflow

elastic_net_workflow <- workflow() %>% 
  add_model(elastic_net_model) %>% 
  add_recipe(interactions_recipe)


elastic_net_stack <- tune_grid(
  elastic_net_workflow, 
  resamples = car_folds, 
  grid = elastic_net_grid, 
  control = control_stack_grid())


# Write out results & workflow

save(elastic_net_stack, file = "results/elastic_net_stack.rda")