### SVM Poly ---

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
svm_poly_model <- svm_poly(
  mode = "classification",
  cost = tune(),
  degree = tune(),
  scale_factor = tune()
) %>%
  set_engine("kernlab")

# set-up tuning grid ----
svm_poly_params <- extract_parameter_set_dials(svm_poly_model)

svm_poly_grid <- grid_regular(svm_poly_params, levels = 5)

# workflow ----
svm_poly_workflow <- workflow() %>% 
  add_model(svm_poly_model) %>% 
  add_recipe(kitchen_sink_recipe)

# fitting
tic.clearlog()
tic("SVM Poly")

svm_poly_tuned <- tune_grid(
  svm_poly_workflow, 
  resamples = car_folds, 
  grid = svm_poly_grid, 
  control = control_grid(save_pred = TRUE, 
                         save_workflow = TRUE, 
                         parallel_over = "everything"))


# Pace tuning code in here
toc(log = TRUE)

# save runtime info
time_log <- tic.log(format = FALSE)

svm_poly_tictoc <- tibble(model = time_log[[1]]$msg, 
                          runtime = time_log[[1]]$toc - time_log[[1]]$tic)

# Write out results & workflow

save(svm_poly_tuned, svm_poly_tictoc, file = "results/svm_poly_tuned.rda")