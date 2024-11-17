# *** Setting Working Directory ***
setwd("C:/Users/marcos/Documents/Projetos_Github/Projetos_DrivenData/Flu_Shot_Learning")
getwd() 

# *** Loading Required Libraries ***
library(tidyverse)  # For data manipulation and visualization
library(ggplot2)    # For advanced plotting
library(corrplot)   # For correlation matrix visualization
library(randomForest)  # For RandomForest model
library(ROCR)       # For ROC curve and AUC calculation
library(mice)       # For imputation (optional step)
library(caret)      # For machine learning workflows

# *** Part 1 - Understanding the Dataset ***

# Loading datasets
df_features <- read_csv("training_set_features.csv")  # Features dataset
df_labels <- read_csv("training_set_labels.csv")      # Labels dataset

# Display dimensions of datasets
dim(df_features)  # Expected dimensions: 26707 x 36
dim(df_labels)    # Expected dimensions: 26707 x 3

# Merging features and labels by respondent_id
merged_df <- merge(df_features, df_labels, by = "respondent_id", all = TRUE)
dim(merged_df)  # Confirm the merged dataset dimensions

# Function to check NA values in a dataset
check_na <- function(x) {
  colSums(is.na(x))
}

# Checking for missing values in merged_df
check_na(merged_df)

# Observation: Most NA values are in "employment_industry" and "employment_occupation".
# Removing the temporary objects to free up memory
rm(df_features, df_labels)

# *** Part 2 - Data Preparation ***

# Converting qualitative variables to factors (except respondent_id)
df_factored <- merged_df %>%
  mutate(across(-respondent_id, as.factor))

# Checking the structure of the factored dataframe
str(df_factored)

# *** Part 3 - Data Exploration ***

# Plotting distributions of target variables
hist(merged_df$h1n1_vaccine,
     breaks = 20,
     col = "lightblue",
     main = "Distribution of H1N1 Vaccine Uptake",
     xlab = "H1N1 Vaccine Uptake (0 = No, 1 = Yes)",
     ylab = "Frequency")
# Observation: The variable is imbalanced.

hist(merged_df$seasonal_vaccine,
     breaks = 20,
     col = "lightblue",
     main = "Distribution of Seasonal Vaccine Uptake",
     xlab = "Seasonal Vaccine Uptake (0 = No, 1 = Yes)",
     ylab = "Frequency")
# Observation: The variable is balanced.

# Removing rows with NA values for further analysis
df_clean <- na.omit(df_factored)

# Observation: Dataset size reduced by approximately 76%. 
# The clean dataset is now 6437 x 36.

# *** Part 4 - Building Random Forest Models ***

# RandomForest for h1n1_vaccine
model_one <- randomForest(
  h1n1_vaccine ~ ., data = df_clean,
  ntree = 100, nodesize = 10, importance = TRUE
)
varImpPlot(model_one)

# RandomForest for seasonal_vaccine
model_two <- randomForest(
  seasonal_vaccine ~ ., data = df_clean,
  ntree = 100, nodesize = 10, importance = TRUE
)
varImpPlot(model_two)

# Observation: The most relevant variables have been identified.

# Selecting variables of interest for further modeling
variables_of_interest <- c(
  "respondent_id", "seasonal_vaccine", "h1n1_vaccine", "doctor_recc_h1n1", 
  "opinion_h1n1_risk", "opinion_h1n1_vacc_effective", "doctor_recc_seasonal", 
  "opinion_seas_risk", "opinion_seas_vacc_effective", "household_adults", 
  "household_children", "employment_occupation", "employment_industry"
)

# Removing unused objects to free memory
rm(model_one, model_two)

# Creating a subset with selected variables
train_df <- merged_df[, variables_of_interest]
str(train_df)

# Removing unused objects to free memory
rm(df_clean, df_factored, merged_df)

# Converting variables to factors (except respondent_id)
train_df <- train_df %>%
  mutate(across(-respondent_id, as.factor))
str(train_df)

# *** Part 5 - Imputation (Optional Step) ***

# Uncomment the following block to apply imputation using mice:
# imputed_data <- mice(train_df, method = 'pmm', m = 5, maxit = 13, seed = 123)
# df_train <- complete(imputed_data)
# write_csv(df_train, "df_train")

# For this example, load preprocessed data
df_train <- read_csv("df_train")
str(df_train)

# Removing unused objects to free memory
rm(train_df)

# Verify missing values after loading preprocessed data
check_na(df_train)

# Converting variables to factors (except respondent_id)
df_train <- df_train %>%
  mutate(across(-respondent_id, as.factor))

# Creating h1n1_train dataset to train the ML model (removing seasonal_vaccine)
h1n1_train <- df_train
h1n1_train$seasonal_vaccine <- NULL
str(h1n1_train)

# Creating seasonal_train dataset to train the ML model (removing h1n1_vaccine)
seasonal_train <- df_train
seasonal_train$h1n1_vaccine <- NULL
str(seasonal_train)

# *** Part 7 - Building Final Models ***

# RandomForest model for h1n1_vaccine
model_h1n1 <- randomForest(
  h1n1_vaccine ~ . , data = h1n1_train, 
  ntree = 500, mtry = 5, importance = TRUE
)

pred_h1n1_train <- predict(model_h1n1, h1n1_train)

# RandomForest model for seasonal_vaccine
model_seasonal <- randomForest(
  seasonal_vaccine ~ ., data = seasonal_train, 
  ntree = 500, mtry = 5, importance = TRUE
)

pred_seasonal_train <- predict(model_seasonal, seasonal_train)

# *** Calculate AUC ***

# Convert true labels to numeric (0 or 1)
true_h1n1 <- as.numeric(h1n1_train$h1n1_vaccine) 
true_seasonal <- as.numeric(seasonal_train$seasonal_vaccine)

# Ensure the predictions are numeric vectors
pred_h1n1_prob <- as.numeric(pred_h1n1_train)
pred_seasonal_prob <- as.numeric(pred_seasonal_train)

# *** Calculate AUC for h1n1_vaccine *** 
pred_h1n1_rocr <- prediction(pred_h1n1_prob, true_h1n1)
auc_h1n1 <- performance(pred_h1n1_rocr, "auc")
print(paste("AUC for h1n1_vaccine:", auc_h1n1@y.values[[1]]))

# ***  Calculate AUC for seasonal_vaccine *** 
pred_seasonal_rocr <- prediction(pred_seasonal_prob, true_seasonal)
perf_seasonal <- performance(pred_seasonal_rocr, "tpr", "fpr")
auc_seasonal <- performance(pred_seasonal_rocr, "auc")
print(paste("AUC for seasonal_vaccine:", auc_seasonal@y.values[[1]]))

# Plot ROC curve for seasonal_vaccine
plot(perf_seasonal, main = "ROC Curve for Seasonal Vaccine", col = "green", lwd = 2)
abline(a = 0, b = 1, col = "red", lty = 2)  # Diagonal random line


# *** Part 8 - Making Predictions and Generating Submission ***

# Load test dataset
teste <- read_csv("test_set_features.csv")
check_na(teste)
str(teste)

# Converting variables to factors (except respondent_id)
teste <- teste %>%
  mutate(across(-respondent_id, as.factor))

# Subsetting test data to match features used in training
teste <- teste[, c("respondent_id", "doctor_recc_h1n1", 
                   "opinion_h1n1_risk", "opinion_h1n1_vacc_effective", 
                   "doctor_recc_seasonal", "opinion_seas_risk", 
                   "opinion_seas_vacc_effective", "household_adults", 
                   "household_children", "employment_occupation", 
                   "employment_industry")]

# Imputing missing values in the test data using mice
#imputed_teste <- mice(teste, method = 'pmm', m = 5, maxit = 13, seed = 123)
#teste <- complete(imputed_teste)
#str(teste)

# Saving the teste file imputed
#write.csv(teste, "teste.csv")

# Loading teste data
teste <- read_csv("teste.csv")

# Making predictions for h1n1_vaccine
pred_h1n1 <- predict(model_h1n1, teste, type = "prob")[, 2]

# Making predictions for seasonal_vaccine
pred_seasonal <- predict(model_seasonal, teste, type = "prob")[, 2]


# Creating a submission dataframe
submission <- data.frame(
  respondent_id = teste$respondent_id,
  h1n1_vaccine = pred_h1n1,
  seasonal_vaccine = pred_seasonal
)

# Saving the submission file
write_csv(submission, "submission.csv")
