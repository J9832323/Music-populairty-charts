############################################################
# Music Popularity Project - R Script
# Dataset: music_dataset.csv
############################################################

############################################################
# 2. PROJECT DESCRIPTION (TEXT IN REPORT)
#
# In the report (not code), write:
# - Context: music industry, streaming platforms, chart success.
# - Business problem: predict Peak Chart Position and identify
#   which attributes (streams, TikTok, danceability, etc.)
#   characterize more popular songs.
# - Goals: accurate prediction + actionable insights for
#   creating/marketing more “hit-like” tracks.
############################################################

set.seed(1)  # for reproducibility
file <- file.choose() 
music <- read.csv(file)

# Quick structure
str(music) #shows the data type of each variable
#summary(music) # might not need this one
dim(music)  # find the dimension of data frame
head(music)  # show the first six rows
View(music)  # show all the data in a new tab

colnames(music)

# Make Genre a factor
#beacuse genre are categories of music need to be treated as such and not as chr
#tells R that these are categories, not text
music$Genre <- as.factor(music$Genre)

# Check missingness
colSums(is.na(music))
#no missing data all values are present

install.packages("tidyverse")
library(tidyverse)

# should we drop "Song" and "Artist"?
music <- music %>% select(-c(Song, Artist))

install.packages("fastDummies")
library(fastDummies)

music <- dummy_cols(music,
                    select_columns = c("Genre"),
                    remove_selected_columns=TRUE,  # remove the original column
                    remove_first_dummy=TRUE)  # removes the first created dummy variable
#music %>% head(2)


#linear regression 
install.packages("caret")
library(caret)

# Set a seed for reproducibility
set.seed(123)

# Split the data into training and testing sets
# 80% for training, 20% for testing
training_indices <- createDataPartition(music$Peak.Position, p = 0.8, list = FALSE)
train_data <- music[training_indices, ]
test_data <- music[-training_indices, ]

# Build the linear regression model
model <- lm(Peak.Position ~ ., data = train_data)
# Summarize the model
summary(model)



# k-NN (k-Nearest Neighbors) Model
set.seed(123) # for reproducibility

# Train the k-NN model using caret's train function
# We'll pre-process the data by centering and scaling, which is important for distance-based algorithms like k-NN.
# We'll also tune the 'k' parameter (number of neighbors) over a range of values.
knn_model <- train(Peak.Position ~ .,
                   data = train_data,
                   method = "knn",
                   preProcess = c("center", "scale"),
                   tuneGrid = expand.grid(k = seq(1, 25, by = 2)), # Try k from 1 to 25 with step 2
                   trControl = trainControl(method = "cv", number = 5)) # 5-fold cross-validation
# Summarize the k-NN model
print(knn_model)

# Plot the results of k-tuning
plot(knn_model)

# Make predictions on the test data
knn_predictions <- predict(knn_model, newdata = test_data)

# Evaluate the model performance
# Calculate RMSE (Root Mean Squared Error)
rmse_knn <- sqrt(mean((knn_predictions - test_data$Peak.Position)^2))
cat("k-NN Model RMSE: ", rmse_knn, "\n")

# Calculate R-squared
r_squared_knn <- cor(knn_predictions, test_data$Peak.Position)^2
cat("k-NN Model R-squared: ", r_squared_knn, "\n")

#Neural Network 
install.packages("neuralnet")
library(neuralnet)


# Prepare the data for the neural network
# Select numerical predictors and the target variable for scaling
# Identify numerical columns (excluding dummy variables which are already 0/1)
numerical_cols <- c("Streams", "Daily.Streams", "Release.Year", "Weeks.on.Chart",
                    "Lyrics.Sentiment", "TikTok.Virality", "Danceability",
                    "Acousticness", "Energy")

# Create a copy of the music data for NN modeling
music_nn <- music

# Store scaling parameters for inverse transformation later
scaled_data_params <- list()

# Scale numerical columns using min-max scaling (0-1 range)
for(col_name in numerical_cols) {
  min_val <- min(music_nn[[col_name]], na.rm = TRUE)
  max_val <- max(music_nn[[col_name]], na.rm = TRUE)
  music_nn[[col_name]] <- (music_nn[[col_name]] - min_val) / (max_val - min_val)
  scaled_data_params[[col_name]] <- list(min = min_val, max = max_val)
}

# Scale the target variable 'Peak.Position' as well
# Important: Store original min/max for inverse scaling predictions
min_peak_position <- min(music_nn$Peak.Position, na.rm = TRUE)
max_peak_position <- max(music_nn$Peak.Position, na.rm = TRUE)
music_nn$Peak.Position <- (music_nn$Peak.Position - min_peak_position) / (max_peak_position - min_peak_position)
scaled_data_params[['Peak.Position']] <- list(min = min_peak_position, max = max_peak_position)

# If not, ensure it's removed to avoid issues with NN model formula
music_nn <- music_nn %>% select(-contains("Genre")) # remove the original Genre column if present

# Identify all predictor variable names for the formula
predictor_names <- setdiff(names(music_nn), c("Peak.Position"))

# Create the neural network formula
formula_nn <- as.formula(paste("Peak.Position ~ ", paste(predictor_names, collapse = " + ")))

# Split data into training and testing sets (70% train, 30% test)
set.seed(123) # for reproducibility
sample_indices_nn <- sample(seq_len(nrow(music_nn)), size = floor(0.70 * nrow(music_nn)))
train_data_nn <- music_nn[sample_indices_nn, ]
test_data_nn <- music_nn[-sample_indices_nn, ]

# Build the Neural Network model
# We'll use a simple network with one hidden layer of 5 neurons as a starting point
nn_model <- neuralnet(formula_nn, data = train_data_nn, hidden = c(5), linear.output = TRUE, stepmax = 1e6)

# Make predictions on the test data
# Extract the predictor columns from the test data
test_predictors <- test_data_nn[, predictor_names]
predictions_scaled <- neuralnet::compute(nn_model, covariate = test_predictors)$net.result

# Inverse scale the predictions to get them back to the original 'Peak.Position' range
predictions <- predictions_scaled * (max_peak_position - min_peak_position) + min_peak_position

# Get actual 'Peak.Position' from test_data and inverse scale them for comparison
actual_peak_position_scaled <- test_data_nn$Peak.Position
actual_peak_position <- actual_peak_position_scaled * (max_peak_position - min_peak_position) + min_peak_position

# Evaluate the model using RMSE
rmse_nn <- sqrt(mean((actual_peak_position - predictions)^2))
cat("Neural Network Root Mean Squared Error (RMSE) for Peak.Position: ", rmse_nn, "\n")

# Plot actual vs predicted values for the Neural Network
plot(actual_peak_position, predictions, main = "Neural Network: Actual vs Predicted Peak Position",
     xlab = "Actual Peak Position", ylab = "Predicted Peak Position",
     col = "darkgreen", pch = 16)
abline(0, 1, col = "red", lwd = 2)


#Ensemble Modeling


#Ensure all models use a consistent training and testing data split (80% train, 20% test) 
#and generate predictions on the same test set, inverse scaling them where necessary.
install.packages("caret")
library(caret)

# Set a seed for reproducibility
set.seed(123)

# Split the data into training and testing sets
# 80% for training, 20% for testing
training_indices <- createDataPartition(music$Peak.Position, p = 0.8, list = FALSE)
train_data <- music[training_indices, ]
test_data <- music[-training_indices, ]
install.packages("neuralnet")
library(neuralnet)

# Prepare the data for the neural network
# Neural networks often perform better with scaled data
# Select numerical predictors and the target variable for scaling

# Identify numerical columns (excluding dummy variables which are already 0/1)
numerical_cols <- c("Streams", "Daily.Streams", "Release.Year", "Weeks.on.Chart",
                    "Lyrics.Sentiment", "TikTok.Virality", "Danceability",
                    "Acousticness", "Energy")

# Create a copy of the music data for NN modeling
music_nn <- music

# Store scaling parameters for inverse transformation later
scaled_data_params <- list()

# Scale numerical columns using min-max scaling (0-1 range)
for(col_name in numerical_cols) {
  min_val <- min(music_nn[[col_name]], na.rm = TRUE)
  max_val <- max(music_nn[[col_name]], na.rm = TRUE)
  music_nn[[col_name]] <- (music_nn[[col_name]] - min_val) / (max_val - min_val)
  scaled_data_params[[col_name]] <- list(min = min_val, max = max_val)
}

# Scale the target variable 'Peak.Position' as well
# Important: Store original min/max for inverse scaling predictions
min_peak_position <- min(music_nn$Peak.Position, na.rm = TRUE)
max_peak_position <- max(music_nn$Peak.Position, na.rm = TRUE)
music_nn$Peak.Position <- (music_nn$Peak.Position - min_peak_position) / (max_peak_position - min_peak_position)
scaled_data_params[['Peak.Position']] <- list(min = min_peak_position, max = max_peak_position)

# Remove original 'Genre' if it still exists (it should be removed by dummy_cols with remove_selected_columns=TRUE)
# If not, ensure it's removed to avoid issues with NN model formula
music_nn <- music_nn %>% select(-contains("Genre")) # remove the original Genre column if present


# Identify all predictor variable names for the formula
predictor_names <- setdiff(names(music_nn), c("Peak.Position"))

# Create the neural network formula
formula_nn <- as.formula(paste("Peak.Position ~ ", paste(predictor_names, collapse = " + ")))

# Apply the consistent training_indices for splitting scaled data
train_data_nn <- music_nn[training_indices, ]
test_data_nn <- music_nn[-training_indices, ]

# Build the Neural Network model
# We'll use a simple network with one hidden layer of 5 neurons as a starting point
nn_model <- neuralnet(formula_nn, data = train_data_nn, hidden = c(5), linear.output = TRUE, stepmax = 1e6)

# Make predictions on the test data
# Extract the predictor columns from the test data
test_predictors <- test_data_nn[, predictor_names]
predictions_scaled <- neuralnet::compute(nn_model, covariate = test_predictors)$net.result

# Inverse scale the predictions to get them back to the original 'Peak.Position' range
predictions <- predictions_scaled * (max_peak_position - min_peak_position) + min_peak_position

# Get actual 'Peak.Position' from test_data and inverse scale them for comparison
actual_peak_position_scaled <- test_data_nn$Peak.Position
actual_peak_position <- actual_peak_position_scaled * (max_peak_position - min_peak_position) + min_peak_position

# Evaluate the model using RMSE
rmse_nn <- sqrt(mean((actual_peak_position - predictions)^2))
cat("Neural Network Root Mean Squared Error (RMSE) for Peak.Position: ", rmse_nn, "\n")

lm_predictions <- predict(model, newdata = test_data)

set.seed(123) # for reproducibility

# Train the k-NN model using caret's train function
# We'll pre-process the data by centering and scaling, which is important for distance-based algorithms like k-NN.
# We'll also tune the 'k' parameter (number of neighbors) over a range of values.
knn_model <- train(Peak.Position ~ .,
                   data = train_data,
                   method = "knn",
                   preProcess = c("center", "scale"),
                   tuneGrid = expand.grid(k = seq(1, 25, by = 2)), # Try k from 1 to 25 with step 2
                   trControl = trainControl(method = "cv", number = 5)) # 5-fold cross-validation

# Make predictions on the test data
knn_predictions <- predict(knn_model, newdata = test_data)

# Evaluate the model performance (optional, but good for verification)
# Calculate RMSE (Root Mean Squared Error)
rmse_knn <- sqrt(mean((knn_predictions - test_data$Peak.Position)^2))
cat("k-NN Model RMSE: ", rmse_knn, "\n")

# Calculate R-squared
r_squared_knn <- cor(knn_predictions, test_data$Peak.Position)^2
cat("k-NN Model R-squared: ", r_squared_knn, "\n")

ensemble_predictions <- (lm_predictions + knn_predictions + predictions) / 3

# Evaluate the ensemble model performance
# Calculate RMSE
rmse_ensemble <- sqrt(mean((ensemble_predictions - test_data$Peak.Position)^2))
cat("Ensemble Model RMSE: ", rmse_ensemble, "\n")

# Calculate R-squared
r_squared_ensemble <- cor(ensemble_predictions, test_data$Peak.Position)^2
cat("Ensemble Model R-squared: ", r_squared_ensemble, "\n")

rmse_lm <- sqrt(mean((lm_predictions - test_data$Peak.Position)^2))
cat("Linear Regression Model RMSE: ", rmse_lm, "\n")

r_squared_lm <- cor(lm_predictions, test_data$Peak.Position)^2
cat("Linear Regression Model R-squared: ", r_squared_lm, "\n")

r_squared_nn <- cor(predictions, actual_peak_position)^2
cat("Neural Network Model R-squared: ", r_squared_nn, "\n")


library(dplyr)

# Create a data frame to store the metrics
metrics_df <- data.frame(
  Model = c("Linear Regression", "k-Nearest Neighbors", "Neural Network", "Ensemble"),
  RMSE = c(rmse_lm, rmse_knn, rmse_nn, rmse_ensemble),
  R_squared = c(r_squared_lm, r_squared_knn, r_squared_nn, r_squared_ensemble)
)

# Print the summary table
print(metrics_df)

