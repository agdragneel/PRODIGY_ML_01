
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

'''
author @Aritra Ghosh

The training and test data were combined at first to do one-hot encoding
and missing value identification.
After that, the data were split again into training attributes and target data.
After fitting the model, the training data was fit.
The training data was predicted, and the data was checked.
The percentage error was calculated, and accuracy was measured.
The test data was then predicted, and appended to test data and saved.
'''

# Read both training and test data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Concatenating training and test data for one-hot encoding
combined_data = pd.concat([train_data, test_data], axis=0, ignore_index=True)

# Handle missing data
combined_data.fillna(0, inplace=True)

# Encode Categorical Values
combined_data = pd.get_dummies(combined_data)

# Separate Attributes and Target Fields
train_X = combined_data.iloc[:len(train_data), :].drop("SalePrice", axis=1)
train_Y = train_data["SalePrice"]
test_X = combined_data.iloc[len(train_data):, :].drop("SalePrice", axis=1)

# Fit Model
model = LinearRegression()
model.fit(train_X, train_Y)

# Predict for Training Data and test data
train_preds = model.predict(train_X)
test_preds = model.predict(test_X)

# Find Error in training predictions and calculate accuracy
train_rmse = np.sqrt(mean_squared_error(train_Y, train_preds))
train_y_mean = train_Y.mean()
percentage_error = ((train_rmse) / train_y_mean) * 100
accuracy = 100 - percentage_error
print("Train RMSE:", train_rmse)
print("Percentage Error:", percentage_error)
print("Accuracy:", accuracy)

# Plotting Training Predictions
plt.figure(figsize=(10, 5))
plt.scatter(train_Y, train_preds, color='blue', label='Training Data')
plt.plot([train_Y.min(), train_Y.max()], [train_Y.min(), train_Y.max()], color='green', linestyle='--')
plt.title('Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()
plt.show()

# Adding Predicted SalesPrice column to test data
test_data["Predicted_SalePrice"] = test_preds
test_data.to_csv("Predicted Data.csv",index=False)