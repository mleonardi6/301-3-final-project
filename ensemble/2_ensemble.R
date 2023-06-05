# Load package(s) ----
library(tidymodels)
library(tidyverse)
library(stacks)

# Handle common conflicts
tidymodels_prefer()

# Load candidate model info ----
result_files <- list.files("ensemble/results/", "*.rda", full.names = TRUE)

result_files

load("data/interactions_recipe.rda")

load("data/car_folds.rda")

for(i in result_files){
  load(i)
}

# Create data stack ----
data_stack <- stacks() %>% 
  add_candidates(boosted_stack) %>% 
  add_candidates(elastic_net_stack) %>% 
  add_candidates(mars_stack) %>% 
  add_candidates(knn_stack) %>% 
  add_candidates(nn_stack)


# Fit the stack ----
# penalty values for blending (set penalty argument when blending)
blend_penalty <- c(10^(-6:-1), 0.5, 1, 1.5, 2)

# Blend predictions using penalty defined above (tuning step, set seed)
set.seed(9876)

stack_blend <- data_stack %>% 
  blend_predictions()

save(stack_blend, file = "ensemble/results/stack_blend.rda")

autoplot(stack_blend)

autoplot(stack_blend, type = "members")

autoplot(stack_blend, type = "weights")

# fit to ensemble to entire training set ----
model_fit <- stack_blend %>% 
  fit_members()

# Save trained ensemble model for reproducibility & easy reference (Rmd report)
save(model_fit, file = "ensemble/results/model_fit.rda")

# load test data
load("data/car_split.rda")

# Explore and assess trained ensemble model
pred <- car_test %>% 
  select(is_claim) %>% 
  bind_cols(predict(model_fit, car_test))

pred_members <- car_test %>% 
  select(is_claim) %>% 
  bind_cols(predict(model_fit, car_test, members = TRUE))

pred_members %>% 
  map_df(roc_auc, truth = is_claim, data = pred_members) %>% 
  mutate(member = colnames(pred_members)) %>% 
  filter(member != "is_claim") %>% 
  arrange(.estimate)

roc_auc(pred, truth = is_claim)

mae(pred, truth = burned, estimate = .pred)

ggplot(pred, aes(x = .pred_class)) +
  geom_bar()

ggplot(pred, aes(x = is_claim)) +
  geom_bar()
