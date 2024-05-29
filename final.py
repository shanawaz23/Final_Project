import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the training and test data
train_data = pd.read_csv('p1_train.csv')
test_data = pd.read_csv('p1_test.csv')

# Rename the columns for better clarity
train_data.columns = ['sensor_1', 'sensor_2', 'health']
test_data.columns = ['sensor_1', 'sensor_2', 'health']

# Split the data into features and target
X_train = train_data[['sensor_1', 'sensor_2']]
y_train = train_data['health']

X_test = test_data[['sensor_1', 'sensor_2']]
y_test = test_data['health']

# Train the Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Train the Support Vector Regression model
svr_model = SVR(kernel='linear')  # Using linear kernel for simplicity
svr_model.fit(X_train, y_train)

# Make predictions on the test set
linear_predictions = linear_model.predict(X_test)
svr_predictions = svr_model.predict(X_test)

# Calculate the Mean Squared Error and Mean Absolute Error for Linear Regression
linear_mse = mean_squared_error(y_test, linear_predictions)
linear_mae = mean_absolute_error(y_test, linear_predictions)

# Calculate the Mean Squared Error and Mean Absolute Error for Support Vector Regression
svr_mse = mean_squared_error(y_test, svr_predictions)
svr_mae = mean_absolute_error(y_test, svr_predictions)

print(f"Linear Regression - MSE: {linear_mse}, MAE: {linear_mae}")
print(f"Support Vector Regression - MSE: {svr_mse}, MAE: {svr_mae}")
