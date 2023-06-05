# get final model results

# load package(s)
library(tidymodels)
library(tidyverse)
library(kableExtra)

tidymodels_prefer()

# load files
load("interactions_recipe/results/knn_tuned_all_interactions.rda")

load("data/interactions_recipe2.rda")

load("data/car_folds.rda")

knn_tuned %>% 
  autoplot(metric = "roc_auc")

# very bad roc_auc