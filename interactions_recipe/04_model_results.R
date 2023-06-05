# get final model results

# load package(s)
library(tidymodels)
library(tidyverse)
library(kableExtra)

tidymodels_prefer()

# load files
result_files <- list.files("interactions_recipe/results/", "*.rda", full.names = TRUE)

result_files

load("data/interactions_recipe.rda")

load("data/car_folds.rda")

for(i in result_files){
  load(i)
}

##########################################
# baseline/null model
null_model <- null_model(mode = "classification") %>% 
  set_engine("parsnip")

null_workflow <- workflow() %>% 
  add_model(null_model) %>% 
  add_recipe(interactions_recipe)

null_fit <- null_workflow %>% 
  fit_resamples(resamples = car_folds,
                control = control_resamples(save_pred = TRUE))

null_fit <- null_fit %>% 
  collect_metrics() %>% 
  mutate(model = "Null") %>% 
  rename(roc_auc = mean) %>% 
  filter(.metric == "roc_auc") %>% 
  select(model, roc_auc)

####################
# put all our tune_grids together
model_set <- as_workflow_set(
  "elastic_net" = elastic_net_tuned, 
  "rand_forest" = rf_tuned,
  "knn" = knn_tuned,
  "boosted_tree" = boosted_tuned,
  "neural_network" = nn_tuned,
  "svm_poly" = svm_poly_tuned,
  "svm_radial" = svm_radial_tuned,
  "mars" = mars_tuned
)

# plot of our results
model_set %>% 
  autoplot(metric = "roc_auc")

# plot just the best models
model_set %>% 
  autoplot(metric = "roc_auc", select_best = TRUE) + 
  theme_minimal() +
  geom_text(aes(y = mean - 0.05, label = wflow_id), angle = 90) +
  ylim(c(0.45, 0.7)) +
  theme(legend.position = "none")

# table of best models
model_results <- model_set %>% 
  group_by(wflow_id) %>% 
  mutate(best = map(result, show_best, metric = "roc_auc", n = 1)) %>% 
  select(best) %>% 
  unnest(cols = c(best))

# computation time
model_times <- bind_rows(elastic_net_tictoc,
                         boosted_tictoc,
                         rf_tictoc,
                         knn_tictoc,
                         nn_tictoc,
                         svm_poly_tictoc,
                         svm_radial_tictoc,
                         mars_tictoc) %>% 
  mutate(wflow_id = c("elastic_net", 
                      "boosted_tree",
                      "rand_forest",
                      "knn",
                      "neural_network",
                      "svm_poly",
                      "svm_radial",
                      "mars"))

results_table <- merge(model_results, model_times) %>% 
  select(model, mean, runtime) %>% 
  rename(roc_auc = mean) 

results_table <- bind_rows(results_table, null_fit) 

# final results table
results_table %>% 
  arrange(desc(roc_auc)) %>% 
  as_tibble() %>% 
  kbl() %>% 
  kable_styling()
