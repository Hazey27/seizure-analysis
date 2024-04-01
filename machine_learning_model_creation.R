# Hazel Dellario
# 2023.11.23
# Trains and tests several different machine learning models on EEG data


#importing
library("rlang")
library("tidyverse")
library("janitor")
library("FactoMineR")
library("factoextra")
library("GGally")
library("corrplot")
library("VGAM")
library("caret")
library("doParallel")
library("randomForest")
library("sjPlot")
library("plotROC")
library("neuralnet")
library("nnet")
library("party")
library("reprtree")


#setwd to source file location
setwd("C:/Users/hazey/OneDrive/Desktop/MLES epilepsy project")


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


#Exploratory Data Analysis --------------------------------
epilepsy_cor <- cor(epilepsy_raw[ , 2:179])

#corrplot(epilepsy_cor, type = "lower") #that is completely uninterpretable 
#ggpairs(epilepsy_raw[ , 2:179]) #not helpful

epilepsy_pca <- PCA(epilepsy_raw[ , 2:179], scale = T)
#well that looks mostly like a circle ngl... Dim 1 only explains 5.61% and dim 2 explains 5.24%

epilepsy_pca$eig
#after 35th dimension, less than 1% variance explained by each dimension
#first 12 dimensions explain 50% of variance 
#first 26 dimensions explain 80% of variance

epilepsy_pca$var$cor
epilepsy_pca$var$cos2
epilepsy_pca$var$contrib #max 1.92 on dim 1; 2.08 on dim 2, 2.32 on dim 3, 2.78 on dim 4, 3.99 on dim 5


fviz_eig(epilepsy_pca) #All dimensions explain about the same amount of variance; dimensional reduction will not be helpful

#fviz_pca_var(epilepsy_pca, axes = c(1, 2), repel = T) #this is not helpful


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

# Calculating t test for every electrode vs the presence of a seizure
for(i in 1:178) {
 epilepsy.t.test[i] <- t.test(epilepsy_raw[ , i + 1] ~ epilepsy_raw$y)
}

#run PCA vs y for each column (too intensive and too much data to be useful....)
# multiPCA <- function(data) {
#   pcaMatrix <- NULL
#   for(i in 1:ncol(data)) {
#     newData[i] <- data.frame()
#     pcaMatrix[i] <- PCA()
#   }
# }


# Basic statistics of EEG values: --------------------------------
# Make sure that individual electrodes don't vary wildly from each other
test_vals <- sample(11385, 7970)

get_mean_sd <- function(var) {
  for(i in 1:length(var)) {
    print(paste("Mean:", mean(apply(epilepsy[var[i], 5:182], MARGIN = 1, FUN = mean)), "SD:", sd(epilepsy[var[i], 5:182])))
  }
}

get_mean_sd(test_vals)

#Values similar enough to compare them en masse

#histogram to gain insight into distribution of data
get_hist <- function(var) {
  for(i in 1:length(var)) {
    hist(t(epilepsy[var[i], 5:182]), main = paste("Histogram of Electrode Values for Trial ", var[i], sep = ""), breaks = 50)
  }
}

get_hist(test_vals) #distributions look very similar, potentially Laplacian

#Looking at a smaller range of electrode values
hist(t(epilepsy[test_vals[1], 5:182]), breaks = 50, xlim = c(-200, 200))

#maintains rough Gaussian shape, may be T distribution
ggplot(data.frame(t(epilepsy[test_vals[1], 5:182])), aes(x = !!sym(paste("X", test_vals[1], sep = "")))) +
  geom_histogram(binwidth = 3, fill = "darkred") +
  stat_function(fun = function(x)
    dnorm(x, mean = mean(t(epilepsy[test_vals[1], 5:182])),
          sd = sd(t(epilepsy[test_vals[1], 5:182])))
    * 4 * length(epilepsy[test_vals[1], 5:182]),
    color = "red", linewidth = 0.8) + #definitely sharper than a normal distribution
  # stat_function(fun = function(x) 
  #   dlaplace(x, location = mean(t(epilepsy[test_vals[1], 5:182])), 
  #         scale = sd(t(epilepsy[test_vals[1], 5:182])/2.2))
  #   * 1.5 * length(epilepsy[test_vals[1], 5:182]),
  #   color = "green", linewidth = 0.8) + #closer than Gaussian but still not very close
  # stat_function(fun = function(x) 
  #   dcauchy(x, location = mean(epilepsy[ , test_vals[1]]), scale = 50)
  #   * 50 * length(epilepsy[ , test_vals[1]]),
  #   color = "green", linewidth = 0.8) + #looks to be best represented by Cauchy distribution with location 0 and scale 50
  xlab("Electrode Value (mV)") +
  ylab("Frequency") +
  ggtitle(paste("Distribution of Electrode Values in Trial ", test_vals[1], sep = ""))


#More data visualization --------------------------------

#Func to select trials
subset <- function(num) {
  new <- epilepsy[((num - 1) * 23 + 1):((num - 1) * 23 + 23), ] %>% 
    pivot_longer(starts_with("X"),
                 names_to = "electrode", 
                 values_to = "value",
                 names_prefix = "X") %>% 
    mutate(electrode = as.integer(electrode))
  return(new)
}

#Trial 1
T1 <- subset(1)

ggplot(T1, aes(x = electrode, y = value, color = Seconds)) +
  geom_point(alpha = 0.8, size = 2) +
  xlab("Number") +
  ylab("Value (µV)") +
  ggtitle("Electrode number vs recorded value for trial 1")+
  ylim(c(-1000, 2000))

#Trial 2
T2 <- subset(2)
ggplot(T2, aes(x = electrode, y = value, color = Seconds)) +
  geom_point(alpha = 0.8, size = 2) +
  xlab("Number") +
  ylab("Value (µV)") +
  ggtitle("Electrode number vs recorded value for trial 2")+
  ylim(c(-1000, 2000))

#First seizure trial
T104 <- subset(103)
ggplot(T104, aes(x = electrode, y = value, color = Seconds)) +
  geom_point(alpha = 0.8, size = 2) +
  xlab("Number") +
  ylab("Value (µV)") +
  ggtitle("Electrode number vs recorded value for Seizure Activity") +
  ylim(c(-1000, 2000))
#MUCH larger range of values during a seizure


#lets start with ML --------------------------------
#Looking for something to detect a new pattern/larger range/diff distribution of data between conditions

#create train and test set; 70/30 split --------------------------------

#must select entire trials for data to be valid
#also want to select equal non-seizure and seizure sets
no.seizure = filter(epilepsy, isSeizure == 0) %>% #369 Trials
  nest(.by = Trial)
  
seizure = filter(epilepsy, isSeizure == 1) %>% 
  nest(.by = Trial)

train.no.seizure.indices <- sample(1:nrow(no.seizure), size = 0.7 * nrow(no.seizure), replace = TRUE)
train.seizure.indices <- sample(1:nrow(seizure), size = 0.7 * nrow(seizure), replace = TRUE)

train.no.seizure <- no.seizure[train.no.seizure.indices, ]
test.no.seizure <- no.seizure[-train.no.seizure.indices, ]

train.seizure <- seizure[train.seizure.indices, ]
test.seizure <- seizure[-train.seizure.indices, ]

#now we can mix seizure and non-seizure data again and unnest
train.nest <- rbind(train.no.seizure, train.seizure)
test.nest <- rbind(test.no.seizure, test.seizure)

train <- unnest(data = train.nest, cols = data)
test <- unnest(data = test.nest, cols = data)


#Basic stats on test vs train data to ensure adequate data splitting --------------------------------

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

#seizures have a much wider range of values as well as a slightly higher mean


#Model training (Random forest) --------------------------------

#must make trial column have acceptable variable names (can't start with a number)
train$isSeizure_chr <- as.factor(ifelse(train$isSeizure == 0, "N", "Y"))
train_nn <- cbind(train[ , 1:2], isSeizure = train$isSeizure_chr, train[ , 5:182])

test$isSeizure_chr <- as.factor(ifelse(test$isSeizure == 0, "N", "Y"))
test_nn <- cbind(test[ , 1:2], isSeizure = test$isSeizure_chr, test[ , 5:182])


# Convert column names to numeric
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
                  tuneLength = 10, #repeating this 10 times (so 5*3*10 times total)
                  trControl = mod.style,
                  linear.output = FALSE)
  stopCluster(cl)
  return(model)
}

start.time <- Sys.time()
# rf_model <- create_model(train_nn[ , 3:181], "rf") 
stop.time <- Sys.time()

rf.runtime <- stop.time - start.time
#saveRDS(rf_model, "epilepsy_rf_model.RDS")
#saving model to file bc i should've done that the first time
#saveRDS(rf_model, file = "epilepsy_rf_model_wo_y.RDS")

rf_model <- readRDS("epilepsy_rf_model.RDS")

#Model1 results----------------
print(rf_model) #best model mtry = 676

rf_var_imp <- varImp(rf_model, scale = F) #variable importance
# X21 best predictor, y2?, y4?, X19
rf_top_preds_list <- slice_max(rf_var_imp[["importance"]], n = 30, order_by = Overall)
rf_top_preds_df <- cbind(predictor = as.ordered(row.names(rf_top_preds_list)),
                         rf_top_preds_list)

rf_top_preds_df$predictor <- factor(rf_top_preds_df$predictor, rf_top_preds_df$predictor, ordered = T)

ggplot(rf_top_preds_df, aes(x = Overall, y = predictor)) +
  geom_col() +
  scale_y_discrete(limits=rev) +
  xlab("Variable Importance") +
  ylab("Predictor") +
  ggtitle("Best Predictors for Random Forest Model")

train$rf_pred <- predict(rf_model, train_nn[ , 3:181])
confusionMatrix(train$rf_pred, train_nn$isSeizure, positive = 'Y')

test$rf_pred <- predict(rf_model, test_nn)
confusionMatrix(test$rf_pred, test_nn$isSeizure, positive = 'Y')
#Accuracy 0.98%;
# Mostly false negatives, so that's not great

reprtree:::plot.reprtree(ReprTree(rf_model, test_nn)) # plot one tree of random forest model
# to illustrate how decision trees work; not relevant to analysis


#Neural network ------------------------------------------------------------------------------

#must first transform binary variable to double
train <- train %>% 
    mutate(isSeizure_int = ifelse(
        isSeizure == 0,
        0,
        1
    ))
test <- test %>% 
    mutate(isSeizure_int = ifelse(
        isSeizure == 0,
        0,
        1
    ))


train$Trial_int <- as.integer(train$Trial)
test$Trial_int <- as.integer(test$Trial)

train$Seconds_int <- as.integer(train$Seconds)
test$Seconds_int <- as.integer(test$Seconds)


create_model <- function(df, model) {
  mod.style <- trainControl(method = "repeatedcv", #repeated cross validation
                            number = 5, #run cross validation 5 times (number of folds)
                            repeats = 3,
                            classProbs = TRUE) #repeat this 3 times (number of complete sets to fold)
  
  cl <- makePSOCKcluster(6) #number of cores to run on in parallel (laptop has 6)
  registerDoParallel(cl) #setting it up so computer runs in parallel
  
  model <-  train(isSeizure ~ .,
                  data = df,
                  method = model,
                  tuneLength = 10, #repeating this 10 times
                  trControl = mod.style,
                  linear.output = FALSE)
  stopCluster(cl)
  return(model)
}

#Keras section #### This code worked on my laptop but not on my desktop, hence why it's commented out
#found other methods to do what I was doing here so not necessary

#library(keras)
# nn_start <- Sys.time()
# # Define a simple neural network with an embedding layer
# keras_nn_model <- keras_model_sequential() %>%
#   layer_embedding(input_dim = 178, output_dim = 178, input_length = 178) %>%
#   layer_flatten() %>%
#   layer_dense(units = 64, activation = 'relu') %>% #hidden layer with 64 neurons, ReLU = rectified linear unit, introduces non-linearity to layers; allows to learn complex patterns
#   layer_dense(units = 1, activation = 'sigmoid') #hidden layer with 1 neuron
# 
# # Compile the model
# keras_nn_model %>% compile(
#   loss = 'binary_crossentropy', #aka log loss, suitable when output is probabilities for binary variable, punishes model for positives away from 1 and negatives away from 0
#   optimizer = optimizer_rmsprop(), #RMSProp = Root mean squared propagation; adaptive learning rate based on past gradients, uses exponentially decaying average of past squared gradients; gives more weight to recent gradients
#   metrics = c('accuracy')
# )

# Train the model

# as.numeric() or similar functions to convert the data if needed.


# train_nn[ , 4:181] <- as.data.frame(sapply(train_nn[ , 4:181], as.numeric))
# keras_nn_model %>% fit(
#   x = train_nn[ , 4:181],
#   y = as.double(train$isSeizure),
#   epochs = 10,
#   batch_size = 32
# )
# nn_end <- Sys.time()

nn_start <- Sys.time() #only takes 3-6 minutes
#nn_model <- neuralnet(isSeizure ~ ., data = train[ , 4:182], hidden = 50, linear.output = FALSE, likelihood = TRUE)
nn_end <- Sys.time()

nn_runtime <- nn_end - nn_start

#saveRDS(nn_model, "epilepsy_nn_model_h_50.RDS")
nn_model <- readRDS("epilepsy_nn_model_h_50.RDS")

nn_start <- Sys.time()
#caret_nn_model <- create_model(train_nn[ , 3:181], "nnet")
nn_end <- Sys.time()

caret_nn_runtime <- nn_end - nn_start

#saveRDS(caret_nn_model, "epilepsy_caret_nn_model.RDS")
caret_nn_model <- readRDS("epilepsy_caret_nn_model.RDS")

train$nn_pred <- predict(nn_model, train[ , 4:182])
test$nn_pred <- predict(nn_model, test[ , 4:182])

ggplot(test, aes(m = nn_pred[, 2], d = isSeizure_int)) +
  geom_roc(n.cuts = 25, labels = F) +
  style_roc(theme = theme_bw) +
  geom_rocci(fill = "pink") +
  geom_abline() +
  ggtitle("ROC Curve for Neural Network")
#ROC curve looks like it's saying that a cutoff of 0.24 is best

apply_cutoff <- function(data, val) {
    cut_col = ifelse(data < val, 0, 1)
    return(cut_col)
}

train$nn_0.45 <- apply_cutoff(train$nn_pred[, 2], 0.45)
test$nn_0.45 <- apply_cutoff(test$nn_pred[, 2], 0.45)
test$nn_0.15 <- apply_cutoff(test$nn_pred, 0.15)

confusionMatrix(as.factor(apply_cutoff(train$nn_pred[, 2], 0.7)), train$isSeizure, positive = "1") #87% accurate
confusionMatrix(as.factor(apply_cutoff(test$nn_pred[, 2], 0.7)), test$isSeizure, positive = "1") #85% accurate


#caret model
train$nn_pred <- predict(caret_nn_model, train_nn[ , 3:181])
test$nn_pred <- predict(caret_nn_model, test_nn[ , 3:181])

confusionMatrix(as.factor(train$nn_pred), train_nn$isSeizure) #87% accurate
confusionMatrix(as.factor(test$nn_pred), test_nn$isSeizure) #85% accurate


# Gradient Boost --------------------------------------------------------

start.time <- Sys.time()
#gbm_model <- create_model()
stop.time <- Sys.time()

logit.runtime <- stop.time - start.time

#saveRDS(gbm_model, file = "epilepsy_gbm_model.RDS")
gbm_model <- readRDS(file = "epilepsy_gbm_model.RDS")

train$gbm_pred <- predict(gbm_model, train[ , c(1:2, 4:182)])
test$gbm_pred <- predict(gbm_model, test[ , c(1:2, 4:182)])

confusionMatrix(train$gbm_pred, train$isSeizure, positive = '1') #100
confusionMatrix(test$gbm_pred, test$isSeizure, positive = '1') #0.94


# Naive Bayes
start.time <- Sys.time()
#naive_bayes_model <- create_model(train_nn[ , 3:181], "naive_bayes")
stop.time <- Sys.time()

#saveRDS(naive_bayes_model, "epilepsy_naive_bayes_model.RDS")
naive_bayes_model <- readRDS("epilepsy_naive_bayes_model.RDS")

train$naive_bayes_preds <- predict(naive_bayes_model, train_nn[ , 3:181])
test$naive_bayes_preds <- predict(naive_bayes_model, test_nn[ , 3:181])

confusionMatrix(train$naive_bayes_preds, train_nn$isSeizure, positive = 'Y')
confusionMatrix(test$naive_bayes_preds, test_nn$isSeizure, positive = 'Y')

#XGBoost Tree
start.time <- Sys.time()
#xgb_model <- create_model(train_nn[ , 3:181], "xgbTree")
stop.time <- Sys.time()

xgb_runtime <- stop.time - start.time

#saveRDS(xgb_model, "epilepsy_xgb_model_model.RDS")
xgb_model <- readRDS("epilepsy_xgb_model_model.RDS")

train$xgb_preds <- predict(xgb_model, train_nn[ , 3:181])
test$xgb_preds <- predict(xgb_model, test_nn[ , 3:181])

confusionMatrix(train$xgb_preds, train_nn$isSeizure, positive = "Y")
confusionMatrix(test$xgb_preds, test_nn$isSeizure, positive = "Y")
