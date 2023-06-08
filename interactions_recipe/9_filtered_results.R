# get final model results

# load package(s)
library(tidymodels)
library(tidyverse)
library(kableExtra)
library(pROC)

tidymodels_prefer()

# load files


load("interactions_recipe/results/boosted_tuned3.rda")

load("data/filtered_interact_recipe.rda")

load("data/filtered_folds.rda")

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

# plot of our results
boosted_tuned_filtered %>% 
  autoplot(metric = "roc_auc")


boosted_model <- boost_tree(mode = "classification", 
                            learn_rate = c(0.005, 0.02), 
                            min_n = tune(),
                            mtry = tune()) %>% 
  set_engine("xgboost", importance = "impurity")

boosted_workflow <- workflow() %>% 
  add_model(boosted_model) %>% 
  add_recipe(interactions_recipe)

boosted_workflow <- boosted_workflow %>% 
  finalize_workflow(select_best(boosted_tuned_filtered, metric = "roc_auc"))

final_fit <- fit(boosted_workflow, filtered_train)

results_prob <- filtered_test %>% 
  bind_cols(predict(final_fit, new_data = filtered_test, type = "prob")) %>% 
  select(c(is_claim, contains("pred")))

results <- filtered_test %>% 
  bind_cols(predict(final_fit, new_data = filtered_test)) %>% 
  select(c(is_claim, contains("pred")))

# create new threshold for prediction probability
new_threshold <- 0.471

# Apply the new threshold to obtain the predicted class labels
predicted_classes <- ifelse(results_prob[, ".pred_Yes"] >= new_threshold, "Yes", "No")

class_counts <- data.frame(Class = predicted_classes)
class_counts <- transform(class_counts, Count = 1)

# Create the bar chart using ggplot
ggplot(class_counts, aes(x = .pred_Yes)) +
  geom_bar() +
  labs(title = "Predicted Classes", x = "Classes", y = "Count")

ggplot(predicted_classes, aes(x = .pred_class, group = is_claim, fill = is_claim)) +
  geom_bar()
