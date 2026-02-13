import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv("sales.csv")

data['Month_Number'] = np.arange(1, len(data) + 1)

X = data[['Month_Number']]
y = data['Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Model Evaluation")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

future_months = np.array(range(len(data) + 1, len(data) + 7)).reshape(-1, 1)
future_sales = model.predict(future_months)

print("\nFuture Sales Forecast:")
for i, sale in enumerate(future_sales, start=1):
    print(f"Month +{i}: {round(sale, 2)}")

plt.figure()
plt.plot(data['Month_Number'], data['Sales'], label='Actual Sales')
plt.plot(data['Month_Number'], model.predict(X), label='Predicted Sales')
plt.plot(future_months, future_sales, '--', label='Future Forecast')
plt.xlabel("Month Number")
plt.ylabel("Sales")
plt.title("Sales & Demand Forecasting")
plt.legend()
plt.show()
