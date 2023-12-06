library(tidymodels)
library(tidyverse)
library(lubridate)
library(vroom)
library(poissonreg)

df_train <- vroom('train.csv', col_types = cols(datetime = col_character()))
df_test <- read_csv('test.csv', col_types = cols(datetime = col_character()))

# Removing other columns not in train data
df_train_clean <- df_train %>%
  select(-casual, -registered) %>%
  select(-holiday, -workingday, -temp) %>%
  mutate(count = log(count))

# Preparing test data
df_test_clean <- df_test %>%
  select(-holiday, -workingday, -temp)


# Feature Engineering 
my_recipe <- recipe(count ~ ., data = df_train_clean) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_mutate(datetime = as_datetime(datetime)) %>%
  step_mutate(hour = hour(datetime)) %>%
  step_mutate(hour = as.factor(hour)) %>%
  step_mutate(season = as.factor(season)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_rm(datetime)
# step_mutate(holiday = as.factor(holiday))
# step_mutate(workingday = as.factor(workingday))


prepped_recipe <- prep(my_recipe)
bake(prepped_recipe, new_data = df_train_clean)


# Training LR Model
my_model <- linear_reg(penalty = tune(), 
                       mixture = tune()) %>%
  set_engine('glmnet')

bike_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_model)
  # fit(data = df_train_clean)

tuning_grid <- grid_regular(penalty(), 
                            mixture(), 
                            levels = 5)

folds <- vfold_cv(df_train_clean, v = 5, repeats = 1)

cv_results <- bike_workflow %>%
  tune_grid(resamples = folds, 
            grid = tuning_grid)

collect_metrics(cv_results) %>%
  filter(.metric=='rmse') %>%
  ggplot(data = ., aes(x = penalty, y = mean, color = factor(mixture))) + 
  geom_line()

best_tune <- cv_results %>%
  select_best("rmse")

final_workflow <- bike_workflow %>%
  finalize_workflow(best_tune) %>%
  fit(data = df_train_clean)

bike_predictions <- predict(final_workflow, new_data = df_test_clean)

# Making Submissions
submission <- df_test %>%
  select(datetime) %>%
  bind_cols(bike_predictions) %>%
  rename(count = .pred) %>%
  mutate(count = exp(count)) %>%
  mutate(count = ifelse(.$count < 0, 0, .$count))

write_csv(submission, 'submission_3.csv')




