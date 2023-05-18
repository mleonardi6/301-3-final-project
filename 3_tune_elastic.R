### Elastic net ----

# Libraries
library(tidyverse)
library(tidymodels)
library(doMC)
library(tictoc)

### Load required objects
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



### Random Forest ---------------

library(tidyverse)
library(tidymodels)
library(doMC)
library(tictoc)

# handle common conflicts
tidymodels_prefer()

# load required objects ----
registerDoMC(cores = 8)
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
  resamples = car_folsd, 
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

### Boosted Tree ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(doMC)
library(tictoc)

# handle common conflicts
tidymodels_prefer()

# load required objects ----
registerDoMC(cores = 8)
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

### KNN ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(doMC)
library(tictoc)

# handle common conflicts
tidymodels_prefer()

# load required objects ----
registerDoMC(cores = 8)
load("data/car_folds.rda")
load("data/car_split.rda")
load("data/kitchen_sink_recipe.rda")

# Define model ----

knn_model <- nearest_neighbor(mode = "classification", 
                              neighbors = tune()) %>% 
  set_engine("kknn")

# set-up tuning grid ----
knn_params <- extract_parameter_set_dials(knn_model)
knn_grid <- grid_regular(knn_params, levels = 5)

# workflow ----
knn_workflow <- workflow() %>% 
  add_model(knn_model) %>% 
  add_recipe(kitchen_sink_recipe)

# fitting
tic.clearlog()
tic("KNN")

knn_tuned <- tune_grid(
  knn_workflow, 
  resamples = car_folds, 
  grid = knn_grid, 
  control = control_grid(save_pred = TRUE, 
                         save_workflow = TRUE, 
                         parallel_over = "everything"))

# Pace tuning code in hear
toc(log = TRUE)

# save runtime info
time_log <- tic.log(format = FALSE)

knn_tictoc <- tibble(model = time_log[[1]]$msg, 
                     runtime = time_log[[1]]$toc - time_log[[1]]$tic)

# Write out results & workflow

save(knn_tuned, knn_tictoc, file = "results/knn_tuned.rda")

### SVM Poly ---

library(tidyverse)
library(tidymodels)
library(doMC)
library(tictoc)


# handle common conflicts
tidymodels_prefer()

# load required objects ----
registerDoMC(cores = 8)
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

### SVM Radial ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(doMC)
library(tictoc)

# handle common conflicts
tidymodels_prefer()

# load required objects ----
registerDoMC(cores = 8)
load("data/car_folds.rda")
load("data/car_split.rda")
load("data/kitchen_sink_recipe.rda")

## define model
svm_radial_model <- svm_rbf(
  mode = "classification", 
  cost = tune(),
  rbf_sigma = tune()
) %>%
  set_engine("kernlab")

# set-up tuning grid ----
svm_radial_params <- extract_parameter_set_dials(svm_radial_model)

svm_radial_grid <- grid_regular(svm_radial_params, levels = 5)

# workflow ----
svm_radial_workflow <- workflow() %>% 
  add_model(svm_radial_model) %>% 
  add_recipe(kitchen_sink_recipe)

# fitting
tic.clearlog()
tic("SVM Radial")

svm_radial_tuned <- tune_grid(
  svm_radial_workflow, 
  resamples =  car_folds, 
  grid = svm_radial_grid, 
  control = control_grid(save_pred = TRUE, 
                         save_workflow = TRUE, 
                         parallel_over = "everything"))

# Pace tuning code in here
toc(log = TRUE)

# save runtime info
time_log <- tic.log(format = FALSE)

svm_radial_tictoc <- tibble(model = time_log[[1]]$msg, 
                            runtime = time_log[[1]]$toc - time_log[[1]]$tic)

# Write out results & workflow

save(svm_radial_tuned, svm_radial_tictoc, file = "results/svm_radial_tuned.rda")

### Neural Network ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(doMC)
library(tictoc)

# handle common conflicts
tidymodels_prefer()

# load required objects ----
registerDoMC(cores = 8)
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
load("data/car_folds.rda")
load("data/car_split.rda")
load("data/kitchen_sink_recipe.rda")

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
  add_recipe(kitchen_sink_recipe)

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

save(mars_tuned, mars_tictoc, file = "results/mars_tuned.rda")

