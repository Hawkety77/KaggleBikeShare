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
  summarise(casual_avg_log = log(mean(casual) + 1),
            registered_avg_log = log(mean(registered) + 1))

#######################
## Predicting Casual ##
#######################

# Removing other columns not in train data
df_train_clean <- df_train %>%
  mutate(casual = log(casual + 1)) %>%
  mutate(datetime = as_datetime(datetime)) %>%
  mutate(hour = as.factor(hour(datetime))) %>%
  mutate(year = as.factor(year(datetime))) %>%
  mutate(month = as.factor(month(datetime))) %>%
  inner_join(hour_avgs, by = 'hour') %>%
  select(-count, -registered, -registered_avg_log)

# Preparing test data
df_test_clean <- df_test %>%
  mutate(datetime = as_datetime(datetime)) %>%
  mutate(hour = as.factor(hour(datetime))) %>%
  mutate(year = as.factor(year(datetime))) %>%
  mutate(month = as.factor(month(datetime))) %>%
  inner_join(hour_avgs, by = 'hour') %>%
  select(-registered_avg_log)

# Feature Engineering 
my_recipe <- recipe(casual ~ ., data = df_train_clean) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_mutate(season = as.factor(season)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_rm(datetime)

prepped_recipe <- prep(my_recipe)
bake(prepped_recipe, new_data = df_train_clean)

my_model_best <- boost_tree(mode = "regression", 
                            trees = 100,
                            tree_depth = 4, 
                            learn_rate = .29) %>%
  set_engine("xgboost")

# Cross Validation

folds <- vfold_cv(df_train_clean, v = 5, repeats = 1)

bike_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_model_best)

fittedfolds <-
  bike_workflow %>%
  fit_resamples(folds)

collect_metrics(fittedfolds)[1, 3]

bike_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_model_best) %>%
  fit(data = df_train_clean)

bike_predictions <- predict(bike_workflow, new_data = df_test_clean)

casual_submission <- df_test %>%
  select(datetime) %>%
  bind_cols(bike_predictions) %>%
  rename(casual = .pred) %>%
  mutate(casual = exp(casual) - 1) %>%
  mutate(casual = ifelse(.$casual < 0, 0, .$casual))

###########################
## Predicting Registered ##
###########################

# Removing other columns not in train data
df_train_clean <- df_train %>%
  mutate(registered = log(registered + 1)) %>%
  mutate(datetime = as_datetime(datetime)) %>%
  mutate(hour = as.factor(hour(datetime))) %>%
  mutate(year = as.factor(year(datetime))) %>%
  mutate(month = as.factor(month(datetime))) %>%
  inner_join(hour_avgs, by = 'hour') %>%
  select(-count, -casual, -casual_avg_log)

# Preparing test data
df_test_clean <- df_test %>%
  mutate(datetime = as_datetime(datetime)) %>%
  mutate(hour = as.factor(hour(datetime))) %>%
  mutate(year = as.factor(year(datetime))) %>%
  mutate(month = as.factor(month(datetime))) %>%
  inner_join(hour_avgs, by = 'hour') %>%
  select(-casual_avg_log)


# Feature Engineering 
my_recipe <- recipe(registered ~ ., data = df_train_clean) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_mutate(season = as.factor(season)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_rm(datetime)

prepped_recipe <- prep(my_recipe)
bake(prepped_recipe, new_data = df_train_clean)

# Cross Validation

folds <- vfold_cv(df_train_clean, v = 5, repeats = 1)

bike_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_model_best)

fittedfolds <-
  bike_workflow %>%
  fit_resamples(folds)

collect_metrics(fittedfolds)[1, 3]

my_model_best <- boost_tree(mode = "regression", 
                            trees = 100,
                            tree_depth = 4, 
                            learn_rate = .29) %>%
  set_engine("xgboost")

bike_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_model_best) %>%
  fit(data = df_train_clean)

bike_predictions <- predict(bike_workflow, new_data = df_test_clean)

registered_submission <- df_test %>%
  select(datetime) %>%
  bind_cols(bike_predictions) %>%
  rename(registered = .pred) %>%
  mutate(registered = exp(registered) - 1) %>%
  mutate(registered = ifelse(.$registered < 0, 0, .$registered))


########################
## Making Submissions ##
########################

submission <- casual_submission %>%
  full_join(registered_submission, by = 'datetime') %>%
  mutate(count = casual + registered) %>%
  select(datetime, count)

write_csv(submission, 'submission_18.csv')





