library(tidyverse)
library(tidymodels)
library(ggplot2)

load("data/car_split.rda")

car_train2 <- car_train %>% 
  select(-policy_id) %>% 
  select(-where(is.factor)) %>% 
  mutate(is_claim = car_train$is_claim)

df_long <- car_train2 %>%
  pivot_longer(cols = -is_claim, names_to = "Variable", values_to = "Value")

# Create scatter plots for each variable against "y"
ggplot(df_long, aes(x = is_claim, y = Value)) +
  geom_jitter() +
  geom_smooth(method = "lm", se = FALSE) +
  facet_wrap(~ Variable, scales = "free")
