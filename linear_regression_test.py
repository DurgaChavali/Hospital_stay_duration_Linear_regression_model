import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Sample Data Creation
data = pd.DataFrame({
    'Age': [25, 45, 65, 35, 50],
    'Weight': [70, 80, 90, 60, 75],
    'Blood_Pressure': [120, 130, 145, 115, 135],
    'Stay_Duration': [2, 5, 7, 3, 4]
})

# Features and Target
X = data[['Age', 'Weight', 'Blood_Pressure']]
y = data['Stay_Duration']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Visualization
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Stay Duration")
plt.ylabel("Predicted Stay Duration")
plt.title("Actual vs Predicted")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red')
plt.show()
