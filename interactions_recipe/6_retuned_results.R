library(tidyverse)
library(tidymodels)

load("interactions_recipe/results/boosted_retuned.rda")

show_best(boosted_retuned, metric = "roc_auc")[1,]

boosted_model <- boost_tree(mode = "classification", 
                            learn_rate = tune(), 
                            min_n = tune(),
                            mtry = tune()) %>% 
  set_engine("xgboost", importance = "impurity")

boosted_workflow <- workflow() %>% 
  add_model(boosted_model) %>% 
  add_recipe(interactions_recipe)

boosted_workflow <- boosted_workflow %>% 
  finalize_workflow(select_best(boosted_retuned, metric = "roc_auc"))

final_fit <- fit(boosted_workflow, car_train)

results_prob <- car_test %>% 
  bind_cols(predict(final_fit, new_data = car_test, type = "prob")) %>% 
  select(c(is_claim, contains("pred")))

results <- car_test %>% 
  bind_cols(predict(final_fit, new_data = car_test)) %>% 
  select(c(is_claim, contains("pred")))


roc_auc(results_prob, truth = is_claim, estimate = .pred_No)

ggplot(results, aes(x = .pred_class, group = is_claim, fill = is_claim)) +
  geom_bar()
