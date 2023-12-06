library(tidymodels)
library(tidyverse)
library(lubridate)
library(vroom)
library(poissonreg)
library(xgboost)
library(rsample)
library(stacks)


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

folds <- vfold_cv(df_train_clean, v = 5, repeats = 2)

untunedModel <- control_stack_grid()
tunedModel <- control_stack_resamples()

# Linear Regression

my_model <- linear_reg() %>%
  set_engine('glm')

bike_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_model)

linreg_folds_fit <- fit_resamples(bike_workflow,
              resamples = folds,
              metrics = metric_set(rmse, mae, rsq),
              control = tunedModel)

# Poisson Regression

my_model <- poisson_reg() %>%
  set_engine('glm')
  
bike_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_model)
  
poireg_folds_fit <- fit_resamples(bike_workflow,
                                  resamples = folds,
                                  metrics = metric_set(rmse, mae, rsq),
                                  control = tunedModel)
  

# Penalized Regression
  
my_model <- linear_reg(penalty = tune(), 
                         mixture = tune()) %>%
            set_engine('glmnet')

bike_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_model)

tuning_grid <- grid_regular(penalty(), 
                            mixture(), 
                            levels = 5)

penreg_folds_fit <- bike_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(rmse, mae, rsq),
            control = untunedModel)


# Trees 

my_model <- decision_tree(tree_depth = tune(), 
                          cost_complexity = tune(), 
                          min_n = tune()) %>%
  set_engine("rpart") %>%
  set_mode("regression")

bike_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_model)

tuning_grid <- grid_regular(tree_depth(), 
                            cost_complexity(), 
                            min_n(), 
                            levels = 3)


trees_folds_fit <- bike_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(rmse, mae, rsq),
            control = untunedModel)


# Random Forest

my_model <- rand_forest(mtry = tune(), 
                        trees = 500, 
                        min_n = tune()) %>%
  set_engine("ranger") %>%
  set_mode("regression")

bike_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_model)

tuning_grid <- grid_regular(mtry(c(3, 32)), 
                                       min_n(), 
                                       levels = 2)

randfor_folds_fit <- bike_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(rmse, mae, rsq),
            control = untunedModel)

# Stacking

bike_stack <- stacks() %>%
  add_candidates(linreg_folds_fit) %>%
  add_candidates(poireg_folds_fit) %>%
  add_candidates(penreg_folds_fit) %>%
  add_candidates(trees_folds_fit) %>%
  add_candidates(randfor_folds_fit)

fitted_bike_stack <- bike_stack %>%
  blend_predictions() %>%
  fit_members()

collect_parameters(fitted_bike_stack, "trees_folds_fit")


# Make Predictions
bike_predictions <- predict(fitted_bike_stack, new_data = df_test_clean)

# Making Submissions
submission <- df_test %>%
  select(datetime) %>%
  bind_cols(bike_predictions) %>%
  rename(count = .pred) %>%
  mutate(count = exp(count)) %>%
  mutate(count = ifelse(.$count < 0, 0, .$count))

write_csv(submission, 'submission_8.csv')




