### Elastic net ----

# Libraries
library(tidyverse)
library(tidymodels)
library(doMC)
library(tictoc)

### Load required objects
registerDoMC(cores = 8)
load("data/car_folds.rda")
load("data/car_split.rda")
load("data/kitchen_sink_recipe.rda")

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
  add_recipe(kitchen_sink_recipe)

### Fitting
tic.clearlog()
tic("Elastic Net")

elastic_net_tuned <- tune_grid(
  elastic_net_workflow, 
  resamples = car_folds, 
  grid = elastic_net_grid, 
  control = control_grid(save_pred = TRUE, 
                         save_workflow = TRUE, 
                         parallel_over = "everything"))

# Pace tuning code in hear
toc(log = TRUE)

# save runtime info
time_log <- tic.log(format = FALSE)

elastic_net_tictoc <- tibble(model = time_log[[1]]$msg, 
                             runtime = time_log[[1]]$toc - time_log[[1]]$tic)

# Write out results & workflow

save(elastic_net_tuned, elastic_net_tictoc, file = "results/elastic_net_tuned.rda")