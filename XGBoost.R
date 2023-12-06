library(tidymodels)
library(tidyverse)
library(lubridate)
library(vroom)
library(poissonreg)
library(xgboost)
library(rsample)


df_train <- vroom('train.csv', col_types = cols(datetime = col_character()))
df_test <- read_csv('test.csv', col_types = cols(datetime = col_character()))

hour_avgs <- df_train %>% 
  mutate(datetime = as_datetime(datetime)) %>%
  mutate(hour = hour(datetime)) %>%
  mutate(hour = as.factor(hour)) %>%
  group_by(hour) %>%
  summarise(count_avg = mean(count))

# Removing other columns not in train data
df_train_clean <- df_train %>%
  select(-casual, -registered) %>%
  mutate(count = log(count)) %>%
  mutate(datetime = as_datetime(datetime)) %>%
  mutate(hour = hour(datetime)) %>%
  mutate(hour = as.factor(hour)) %>%
  inner_join(hour_avgs, by = 'hour')

# Preparing test data
df_test_clean <- df_test %>%
  mutate(datetime = as_datetime(datetime)) %>%
  mutate(hour = hour(datetime)) %>%
  mutate(hour = as.factor(hour)) %>%
  inner_join(hour_avgs, by = 'hour')


# Feature Engineering 
my_recipe <- recipe(count ~ ., data = df_train_clean) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_mutate(season = as.factor(season)) %>%
  step_mutate(holiday = as.factor(holiday)) %>%
  step_mutate(workingday = as.factor(workingday)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_rm(datetime)


prepped_recipe <- prep(my_recipe)
bake(prepped_recipe, new_data = df_train_clean)


# Training LR Model
my_model <- boost_tree(mode = "regression", 
                       trees = tune(),
                       tree_depth = tune(),
                       learn_rate = tune(),
                       loss_reduction = tune(),
                       sample_size = tune()
                       ) %>%
  set_engine("xgboost")

# param_grid <- expand.grid(
#   trees = c(100, 200, 300),
#   tree_depth = c(3, 5, 7),
#   learn_rate = c(0.01, 0.05, 0.1),
#   loss_reduction = c(0, 0.1, 0.2),
#   sample_size = c(0.7, 0.8, 0.9)
# )

param_grid <- expand.grid(
  trees = c(100, 200, 300),
  tree_depth = c(3, 5, 7),
  learn_rate = c(0.01, 0.05, 0.1),
  loss_reduction = c(0, 0.1, 0.2),
  sample_size = c(0.7, 0.8, 0.9)
)

xgb_tune <- tune_grid(
  my_model, 
  my_recipe, 
  vfold_cv(df_train_clean, v = 5, repeats = 2), 
  control = control_grid(save_workflow = TRUE), 
  grid = param_grid
)

collect_metrics(xgb_tune)
show_best(xgb_tune, metric = 'rmse')

my_model_best <- boost_tree(mode = "regression", 
                                     trees = 300,
                                     tree_depth = 7,
                                     learn_rate = .1,
                                     loss_reduction = .2,
                                     sample_size = .7
) %>%
  set_engine("xgboost")

bike_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_model_best) %>%
  fit(data = df_train_clean)

bike_predictions <- predict(bike_workflow, new_data = df_test_clean)

# Making Submissions
submission <- df_test %>%
  select(datetime) %>%
  bind_cols(bike_predictions) %>%
  rename(count = .pred) %>%
  mutate(count = exp(count)) %>%
  mutate(count = ifelse(.$count < 0, 0, .$count))

write_csv(submission, 'submission_10.csv')




