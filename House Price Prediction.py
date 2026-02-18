import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Price'] = housing.target

X = df.drop('Price', axis=1).values
y = df['Price'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Info of the dataset:")
print(df.info())
print("Descripe of the dataset:")
print(df.describe())
print("First 5 rows of the dataset:")
print(df.head())

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("=== Improved House Price Prediction ===")
print("Root Mean Squared Error:", round(rmse, 2))
print("R² Score:", round(r2, 2))

plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred, alpha=0.6, color='#1f77b4')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices (Random Forest)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

importances = model.feature_importances_
plt.figure(figsize=(8,5))
plt.bar(df.drop('Price', axis=1).columns, importances, color='#1f77b4')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance in Random Forest')
plt.tight_layout()
plt.show()
