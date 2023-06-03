### Cleaning

library(tidyverse)
library(tidymodels)
library(naniar)

### Load the original data 
car_data <- read_csv("data/train.csv") %>% 
  mutate(
    is_claim = factor(is_claim, levels = c(0, 1), labels = c("No", "Yes")),
    area_cluster = as.factor(area_cluster),
    segment = as.factor(segment),
    model = as.factor(model),
    fuel_type = as.factor(fuel_type),
    max_torque = as.factor(max_torque),
    max_power = as.factor(max_power),
    engine_type = as.factor(engine_type),
    is_esc = as.factor(is_esc),
    is_adjustable_steering = as.factor(is_adjustable_steering),
    is_tpms = as.factor(is_tpms),
    is_parking_sensors = as.factor(is_parking_sensors),
    is_parking_camera = as.factor(is_parking_camera),
    rear_brakes_type = as.factor(rear_brakes_type),
    transmission_type = as.factor(transmission_type),
    steering_type = as.factor(steering_type),
    is_front_fog_lights = as.factor(is_front_fog_lights),
    is_rear_window_wiper = as.factor(is_rear_window_wiper),
    is_rear_window_washer = as.factor(is_rear_window_washer),
    is_rear_window_defogger = as.factor(is_rear_window_defogger),
    is_brake_assist = as.factor(is_brake_assist),
    is_power_door_locks = as.factor(is_power_door_locks),
    is_central_locking = as.factor(is_central_locking),
    is_power_steering = as.factor(is_power_steering),
    is_driver_seat_height_adjustable = as.factor(is_driver_seat_height_adjustable),
    is_day_night_rear_view_mirror = as.factor(is_day_night_rear_view_mirror),
    is_ecw = as.factor(is_ecw),
    is_speed_alert = as.factor(is_speed_alert)
    ### need to do this kind of thing for all non numeric variables in the dataset
  )

### Split data once to cut down dataset with stratifying
set.seed(111)

set_up_split <- initial_split(car_data, prop = .8, strata = is_claim)
discard <- training(set_up_split)
keep <- testing(set_up_split)

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