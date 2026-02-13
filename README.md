# Task 1 - Sales Forecasting using Linear Regression

## 📌 Project Description
This project predicts future sales using Machine Learning.  
Linear Regression model is used to forecast upcoming months' sales based on past sales data.

The model also evaluates performance using different evaluation metrics and shows results in graphical format.

---

## 🎯 Objective
- Load sales dataset
- Convert month data into numerical values
- Train Linear Regression model
- Evaluate model performance
- Predict future sales
- Visualize actual vs predicted sales

---

## 📊 Dataset
Dataset contains monthly sales data.

### Columns:
- Month → Month name
- Sales → Sales amount

---

## 🛠 Libraries Used
- pandas → Data handling
- numpy → Numerical operations
- matplotlib → Data visualization
- scikit-learn → Machine learning model

---

## ⚙ Process / Steps
1. Load dataset using pandas
2. Create Month_Number feature
3. Split data into training and testing
4. Train Linear Regression model
5. Evaluate model using MAE, RMSE, R2 Score
6. Predict future sales for next 6 months
7. Plot graph for visualization

---

## 📈 Output
- Model accuracy metrics
- Future sales prediction
- Graph showing actual vs predicted sales

---

## ▶ How to Run

1. Install required libraries:
pip install pandas numpy matplotlib scikit-learn

2. Run program:
python sales_forecast.py

---

## 📂 Files Included
- sales_forecast.py → Python code
- sales.csv → Dataset
- README.md → Project documentation

---

## 🚀 Future Improvement
- Use more data for better prediction
- Try advanced models
- Deploy as web application
