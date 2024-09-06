import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
import math
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Set the backend for Matplotlib
plt.switch_backend('agg')  # Use 'agg' for non-GUI backend, useful for script-based environments

# Load dataset
file_path = '/Users/vincenzosilvestri/Desktop/DATASET.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Basic dataset information
print(df.head())
print(df.shape)
print(df.info())

# Check and handle duplicates
if df.duplicated().sum() > 0:
    df = df.drop_duplicates()

# Check and handle missing values
if df.isnull().values.any():
    df = df.dropna()  # For simplicity, dropping missing values, could use imputation if needed

# Splitting the dataset into features and target
X = df.drop(columns=['Date', 'Close'])
y = df["Close"]

# Normalizing the features using Z-Score Standardization
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Convert normalized features back to DataFrame
X_normalized_df = pd.DataFrame(X_normalized, columns=X.columns)

# Combine the normalized features with the target variable and Date for final dataset
final_df = pd.concat([df['Date'].reset_index(drop=True), X_normalized_df, y.reset_index(drop=True)], axis=1)

# Save the final normalized dataset to a CSV file
final_df.to_csv('/Users/vincenzosilvestri/Desktop/DATASET_normalized.csv', index=False)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Building the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)

# Evaluating the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
r2 = r2_score(y_test, y_pred)


# Calculate adjusted R²
n = X_test.shape[0]  # number of samples
p = X_test.shape[1]  # number of predictors
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)



print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")
print(f"Adjusted R²: {adjusted_r2}")
# Adding Statsmodels for detailed analysis
X_train_sm = sm.add_constant(X_train)
model_sm = sm.OLS(y_train, X_train_sm).fit()
print(model_sm.summary())

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.grid(True)
plt.savefig('Actual vs Predicted')
plt.close()

# Residuals calculation
residuals = y_test - y_pred

# 1. Residuals vs Fitted Values Plot
plt.figure(figsize=(10,6))
plt.scatter(y_pred, residuals)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')
plt.savefig('residuals_vs_fitted.png')
plt.close()

# 2. Histogram of Residuals
plt.figure(figsize=(10,6))
sns.histplot(residuals, kde=True)
plt.title('Histogram of Residuals')
plt.savefig('histogram_residuals.png')
plt.close()

# 3. Q-Q Plot
plt.figure(figsize=(10,6))
qqplot(residuals, line ='45')
plt.title('Q-Q Plot')
plt.savefig('qqplot_residuals.png')
plt.close()

# 4. ACF of Residuals
plt.figure(figsize=(10,6))
sm.graphics.tsa.plot_acf(residuals, lags=40)
plt.title('ACF of Residuals')
plt.savefig('acf_residuals.png')
plt.close()

# 5. Cook's Distance Plot
influence = model_sm.get_influence()
(c, p) = influence.cooks_distance
plt.figure(figsize=(10,6))
plt.stem(np.arange(len(c)), c, markerfmt=",")
plt.title("Cook's Distance")
plt.savefig('cooks_distance.png')
plt.close()

# Inform the user where the file has been saved
print("Normalized dataset has been saved to '/Users/vincenzosilvestri/Desktop/DATASET_normalized.csv'")
