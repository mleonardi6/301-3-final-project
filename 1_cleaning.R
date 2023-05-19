### Cleaning

library(tidyverse)
library(tidymodels)
library(naniar)

### Load the original data 
car_data <- read_csv("data/train.csv") %>% 
  mutate(
    is_claim = factor(is_claim, levels = c(0, 1), labels = c("no", "yes"))
    ### need to do this kind of thing for all non numeric variables in the dataset
  )

### Split data once to cut down dataset with stratifying
set.seed(111)

set_up_split <- initial_split(car_data, prop = .8, strata = is_claim)
discard <- training(car_split)
keep <- testing(car_split)

## Split again for an actual training and testing set

car_split <- initial_split(keep, prop = .7, strata = is_claim)
car_train <- training(car_split)
car_test <- testing(car_split)

save(car_train, car_test, file = "data/car_split.rda")

### EDA
gg_miss_var(car_data)

# For group: 
# not sure if we need to do a bunch of mutations when we load the data
# to make variables factors
ggplot(car_data, aes(x = is_claim, fill = is_claim, group = is_claim)) +
  geom_bar(aes(fill = is_claim))

ggplot(car_data, aes(x = is_front_fog_lights, fill = is_front_fog_lights, group = is_front_fog_lights)) +
  geom_bar(aes(fill = is_front_fog_lights))
