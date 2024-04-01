# Code from desktop on 12.04.2023
# edited to correct path to file location and increased number of cores to use to 6

#importing
library("tidyverse")
library("sjPlot")
library("plotROC")
library("brms")
library("rstan")

#recommendations to run from rstan
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

#setwd to source file location
setwd("/home/dellah/Bayesian_data_analysis/epilepsy_data_csv/")

epilepsy_raw <- read.csv("epilepsy_data_from_data.world.csv")
#glimpse(epilepsy_raw)

# y contains the category of the 178-dimensional input vector. Specifically y in {1, 2, 3, 4, 5}:
# 5 - eyes open, means when they were recording the EEG signal of the brain the patient had their eyes open
# 4 - eyes closed, means when they were recording the EEG signal the patient had their eyes closed
# 3 - Yes they identify where the region of the tumor was in the brain and recording the EEG activity from the healthy brain area
# 2 - They recorded the EEG from the area where the tumor was located
# 1 - Recording of seizure activity

#each column is a different location where EEG data is being collected
#each row is electrical activity over 1 second


#Pre-processing ---------------------------------------------------------------------

epilepsy <- epilepsy_raw %>% 
  separate(col = X, into = c("Seconds", "Version", "Trial")) %>% 
  na.omit() %>% 
  select(!Version) %>%
  mutate(isSeizure = as.factor(ifelse(y == 1, 1, 0)))

epilepsy$Trial <- replace(
  epilepsy$Trial, 
  which(is.na(epilepsy$Trial)),
  value = -1                   # removes all rows with Trial val N/A 
)

epilepsy$Trial <- as.double(epilepsy$Trial)


epilepsy$Seconds <- as.double(
  gsub("X", "", epilepsy$Seconds)
)

epilepsy <- epilepsy %>%
  relocate("Trial", "Seconds", "y", "isSeizure") %>% 
  arrange(Trial, Seconds) %>% 
  mutate(y = as.factor(y),
         Trial = as.factor(Trial))


#create train/validate/test set; 50/20/30 split --------------------------------

#must select entire trials for data to be valid
no.seizure = filter(epilepsy, isSeizure == 0) %>%
  nest(.by = Trial)

seizure = filter(epilepsy, isSeizure == 1) %>% 
  nest(.by = Trial)

train.no.seizure.indices <- sample(1:nrow(no.seizure), size = 0.5 * nrow(no.seizure), replace = TRUE)
train.seizure.indices <- sample(1:nrow(seizure), size = 0.5 * nrow(seizure), replace = TRUE)

no.seizure.leftover <- no.seizure[c(-train.no.seizure.indices), ]
seizure.leftover <- seizure[c(-train.seizure.indices), ]

validate.no.seizure.indices <- sample(1:nrow(no.seizure.leftover), size = 0.4 * nrow(no.seizure.leftover), replace = TRUE)
validate.seizure.indices <- sample(1:nrow(seizure.leftover), size = 0.4 * nrow(seizure.leftover), replace = TRUE)

train.no.seizure <- no.seizure[train.no.seizure.indices, ]
train.seizure <- seizure[train.seizure.indices, ]

validate.no.seizure <- no.seizure.leftover[validate.no.seizure.indices, ]
validate.seizure <- seizure.leftover[validate.seizure.indices, ]

#bc replace = TRUE, train ends up being 0.39, validation 0.19, and test 0.41
test.no.seizure <- no.seizure.leftover[-validate.no.seizure.indices, ]
test.seizure <- seizure.leftover[-validate.seizure.indices, ]

#now we can mix seizure and non-seizure data again and unnest
train.nest <- rbind(train.no.seizure, train.seizure)
validate.nest <- rbind(validate.no.seizure, validate.seizure)
test.nest <- rbind(test.no.seizure, test.seizure)

train <- unnest(data = train.nest, cols = data)
validate <- unnest(data = validate.nest, cols = data)
test <- unnest(data = test.nest, cols = data)

#must make trial column have acceptable variable names (can't start with a number)
train$isSeizure_chr <- as.factor(ifelse(train$isSeizure == 0, "N", "Y"))
train_nn <- cbind(train[ , 1:2], isSeizure = train$isSeizure_chr, train[ , 5:182])

test$isSeizure_chr <- as.factor(ifelse(test$isSeizure == 0, "N", "Y"))
test_nn <- cbind(test[ , 1:2], isSeizure = test$isSeizure_chr, test[ , 5:182])


# # Convert column names to numeric
names(train_nn) <- c("Trial", "Seconds", "isSeizure", as.numeric(gsub("^X(\\d+)$", "\\1", colnames(train_nn[ , 4:181]))))
names(test_nn) <- c("Trial", "Seconds", "isSeizure", as.numeric(gsub("^X(\\d+)$", "\\1", colnames(test_nn[ , 4:181]))))


create_model <- function(data, model) {
  mod.style <- trainControl(method = "repeatedcv", #repeated cross validation
                            number = 5, #run cross validation 5 times (number of folds)
                            repeats = 3,
                            classProbs = TRUE) #repeat this 3 times (number of complete sets to fold)
  
  cl <- makePSOCKcluster(6) #number of cores to run on in parallel (laptop has 6)
  registerDoParallel(cl) #setting it up so computer runs in parallel
  
  model <-  train(isSeizure ~ .,
                  data = data,
                  method = model,
                  tuneLength = 10, #repeating this 10 times
                  trControl = mod.style,
                  linear.output = FALSE)
  stopCluster(cl)
  return(model)
}


#Basic stats --------------------------------

#function
stat <- function(data, desired.val = -1) {
  if(desired.val == 0 | desired.val == 1) {
    df = data %>% 
      filter(isSeizure == desired.val)
  } else {
    df = data
  }
  mean = mean(apply(df[ , 5:182], 2, FUN = mean))
  median = median(apply(df[ , 5:182], 2, FUN = median))
  range = range(df[ , 5:182])
  return(rbind(mean, median, range))
}

#overall stats
stat(train)

#no seizure stats
stat(train, 0)

#seizure stats
stat(train, 1)


# Hierarchical Normal prior --------------------------------------------------------------------------------

#I should put data into long format so I don't have 178 separate columns
pivot_EEG_longer <- function(data) {
  data_long <- pivot_longer(data, cols = colnames(data[ , 5:182]), names_to = "Electrode", values_to = "EEG_value")
  data_long$Electrode <- factor(data_long$Electrode, levels = colnames(train[ , 5:182]), ordered = TRUE)
  return(data_long)
}

train_long <- pivot_EEG_longer(train)
validate_long <- pivot_EEG_longer(validate)
test_long <- pivot_EEG_longer(test)

train_long$EEG_value <- as.numeric(train_long$EEG_value)
train_long$isSeizure_num <- as.numeric(train_long$isSeizure) - 1

# Setting priors
rs_priors <- c(
  set_prior("normal(0, 100)", class = "Intercept"),  #                Prior for global intercept
  set_prior("normal(0, 1000)", class = "sd"),        #                Prior for global standard deviation
  set_prior("normal(0, 1000)", class = "sd", group = "Trial"),  #      Prior for sd tau (Trial)
  set_prior("normal(0, 1000)", class = "sd", group = "Electrode")   #  Prior for sd eta (Electrode)
) # Dr. Radev says priors are not necessary

# Formula
rs_formula <- bf(
  #isSeizure ~ EEG_value + (EEG_Values | Trial) + (EEG_Values | Electrode),  
  isSeizure_num ~ EEG_value + (1 | Trial) + (1 | Electrode),
  nl = FALSE  #non-linear distribution (normal in this case)
)

start.time = Sys.time()
fit_rs = brm(
  formula = rs_formula,
  data = train_long[ , c(1, 6:8)],
  family = bernoulli(), #bernoulli bc isSeizure is binary; brms applies sigmoid function automatically according to documentation
  #prior = rs_priors,
  iter = 2000,
  warmup = 500,
  chains = 4,
  cores = 6,
  file = "rs_epilepsy_model"
)
end.time = Sys.time()
