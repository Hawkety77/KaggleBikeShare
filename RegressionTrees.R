library(tidymodels)
library(tidyverse)
library(lubridate)
library(vroom)
library(poissonreg)
library(xgboost)
library(rsample)


df_train <- vroom('train.csv', col_types = cols(datetime = col_character()))
df_test <- read_csv('test.csv', col_types = cols(datetime = col_character()))

# Removing other columns not in train data
df_train_clean <- df_train %>%
  select(-casual, -registered) %>%
  select(-temp) %>%
  mutate(count = log(count))

# Preparing test data
df_test_clean <- df_test %>%
  select(-temp)


# Feature Engineering 
my_recipe <- recipe(count ~ ., data = df_train_clean) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_mutate(datetime = as_datetime(datetime)) %>%
  step_mutate(hour = hour(datetime)) %>%
  step_mutate(hour = as.factor(hour)) %>%
  step_mutate(season = as.factor(season)) %>%
  step_mutate(holiday = as.factor(holiday)) %>%
  step_mutate(workingday = as.factor(workingday)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_rm(datetime)


prepped_recipe <- prep(my_recipe)
bake(prepped_recipe, new_data = df_train_clean)


# Training LR Model
my_model <- decision_tree(tree_depth = tune(), 
                          cost_complexity = tune(), 
                          min_n = tune()) %>%
  set_engine("rpart") %>%
  set_mode("regression")

# param_grid <- expand.grid(
#   tree_depth = c(3, 5, 7),
#   cost_complexity = c(.01, .1, 1, 10), 
#   min_n = c(3, 5, 7)
# )

param_grid <- grid_regular(tree_depth(), 
                           cost_complexity(), 
                           min_n(), 
                           levels = 3)

tree_tune <- tune_grid(
  my_model, 
  my_recipe, 
  vfold_cv(df_train_clean, v = 5, repeats = 2), 
  control = control_grid(save_workflow = TRUE), 
  grid = param_grid
)

collect_metrics(tree_tune)
show_best(tree_tune, metric = 'rmse')

my_model_best <- decision_tree(tree_depth = 15, 
                               cost_complexity = .0, 
                               min_n = 21) %>%
  set_engine("rpart") %>%
  set_mode("regression")

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

write_csv(submission, 'submission_6.csv')




