import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from Regression.model import MyRegression

# Load the California housing dataset
data =fetch_california_housing()

# Split the dataset into features and target variable
features=pd.DataFrame(data.data, columns=data.feature_names)
target=data.target
print(features.head())

# Define the test size
test_size = [0.2, 0.3, 0.4, 0.5]
mse_arr = []
r2_arr = []
train_size_arr = []

# Iterate over different test sizes
for size in test_size:

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=size, random_state=42)


    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    # Initialize and train the regression model
    model = MyRegression()  
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    train_size_arr.append(X_train.shape[0])

    # Evaluate the model
    mse, r2 = model.evaluate(y_test, y_pred)
    mse_arr.append(mse)
    r2_arr.append(r2)

    # Print the results
    print(f"Test Size: {size}:{1-size}")
    print(f"Mean Squared Error : {mse}")
    print(f"R^2 Score : {r2}")

    # Plotting the predictions vs actual values
    plt.figure(figsize=(10, 6))
    plt.plot(y_test[:100], alpha=0.5,label='Actual Values', color='green')
    plt.plot(y_pred[:100], alpha=0.5, label='Predicted Values', color='orange')
    plt.legend()
    plt.title('Actual vs Predicted Values for Test Size ' + str(size))
    plt.show()

# Plotting MSE and R2 against train size
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)    
plt.plot(train_size_arr, mse_arr, marker='o')
plt.title('Mean Squared Error vs Train Size')
plt.xlabel('Train Size')
plt.ylabel('Mean Squared Error')
plt.subplot(1, 2, 2)
plt.plot(train_size_arr, r2_arr, marker='o', color='orange')
plt.title('R^2 Score vs Train Size')
plt.xlabel('Train Size')
plt.ylabel('R^2 Score')
plt.tight_layout()
plt.show()