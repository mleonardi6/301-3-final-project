### Cleaning

library(tidyverse)
library(tidymodels)

### Load the original data and merge it into one file----------------------

test <- read_csv("raw_data/test.csv")
train <- read_csv("raw_data/train.csv")

car_raw <- test %>% 
  bind_rows(train)

# save output
save(car_raw, file = "raw_data/car_raw.rda")

### Initial split---------------------------

set.seed(1234)

car_split <- initial_split(car_raw, prop = 0.8, strata = is_claim)

car_train <- training(car_split)
car_test <- testing(car_split)

### Poking around for the data memo---------------------------
