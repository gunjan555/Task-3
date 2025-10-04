# task3_linear_regression.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================
# Load Dataset
# ============================
df = pd.read_csv("dataset.csv")

print("Dataset Preview (before encoding):")
print(df.head())

# ============================
# Encode Categorical Variables
# ============================
df = pd.get_dummies(df, drop_first=True)

print("\nDataset Preview (after encoding):")
print(df.head())

# ============================
# Define Features & Target
# ============================
X = df.drop("price", axis=1)
y = df["price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================
# Train Model
# ============================
model = LinearRegression()
model.fit(X_train, y_train)

# ============================
# Predictions & Evaluation
# ============================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# ============================
# Save Metrics & Coefficients
# ============================
with open("task3_output.txt", "w", encoding="utf-8") as f:
    f.write("Linear Regression Results\n")
    f.write("========================\n")
    f.write(f"MAE: {mae:.4f}\n")
    f.write(f"MSE: {mse:.4f}\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"R² Score: {r2:.4f}\n\n")

    f.write("Intercept (b0): {:.4f}\n\n".format(model.intercept_))
    f.write("Coefficients:\n")
    coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
    f.write(coeff_df.to_string())  # nicely formatted

print("\n✅ Results saved to task3_output.txt")

# ============================
# Plot Actual vs Predicted
# ============================
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color="blue")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linestyle="--")
plt.savefig("task3_plot.png")
plt.close()

print("✅ Regression plot saved as task3_plot.png")