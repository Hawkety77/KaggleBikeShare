## libraries
library(tidyverse)
library(vroom)
library(patchwork)
library(DataExplorer)
library(skimr)
library(lubridate)

## Read in the data
bike_train <- read_csv('train.csv')
bike_test <- read_csv('test.csv')
sample_submission <- read_csv('sampleSubmission.csv')

## Check for missing values
skim(bike_train)

## Checking data types
glimpse(bike_train)

## Assign Factors
bike_train_factored <- bike_train %>%
  mutate(season = as.factor(season)) %>%
  mutate(holiday = as.factor(holiday)) %>%
  mutate(workingday = as.factor(workingday)) %>%
  mutate(hour = hour(datetime)) %>%
  mutate(day = wday(datetime)) %>%
  mutate(hour = as.factor(hour))
  
## Correlation plot
plot_correlation(bike_train_factored)

## Temperature scatter plot
ggplot(data = bike_train_factored) +
  geom_point(aes(temp, count), alpha = .1, color = 'blue')

## Season plot
ggplot(data = bike_train_factored) + 
  geom_col(aes(season, count, color = season))

## Weather Plot
ggplot(data = bike_train_factored) + 
  geom_boxplot(aes(weather, count, group = weather))

## Hour Plot
ggplot(data = bike_train_factored) + 
  geom_col(aes(hour, count/455))

ggplot(data = bike_train_factored) + 
  geom_bar(aes(hour))

bike_train_factored %>%
  group_by(hour) %>%
  summarise(count = n())

ggplot(data = filter(bike_train_factored, day == 6, hour == 5)) + 
  geom_point(aes(x = datetime, y = registered)) +
  ylim(c(0, 300))
  # xlim(as_datetime('2011-01-01'), as_datetime('2011-03-31'))



