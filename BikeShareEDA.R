## libraries
library(tidyverse)
library(vroom)
library(patchwork)
library(DataExplorer)
library(skimr)

## Read in the data
bike_train <- read_csv('train.csv')
bike_test <- read_csv('test.csv')
sample_submission <- read_csv('sampleSubmission.csv')

## View data
view(bike_train)
view(bike_test)

## Check for missing values
skim(bike_train)

## Checking data types
glimpse(bike_train)

## Assign Factors
bike_train_factored <- bike_train %>%
  mutate(season = as.factor(season)) %>%
  mutate(holiday = as.factor(holiday)) %>%
  mutate(workingday = as.factor(workingday))
  
## Correlation plot
plot1 <- plot_correlation(bike_train_factored)

## Temperature scatter plot
plot2 <- ggplot(data = bike_train_factored) +
  geom_point(aes(temp, count), alpha = .1, color = 'blue')

## Season plot
plot3 <- ggplot(data = bike_train_factored) + 
  geom_col(aes(season, count, color = season))

## Weather Plot
plot4 <- ggplot(data = bike_train_factored) + 
  geom_boxplot(aes(weather, count, group = weather))

## Design 4 panel plot

(plot1 + plot2) / (plot3 + plot4)




