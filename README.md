# 📈 Custom Linear Regression (from Scratch)
A clean,   from-scratch implementation of **Linear Regression** using only core Python and NumPy.  
This project demonstrates the fundamental mathematics and logic behind regression models without relying on high-level libraries like `scikit-learn`. It includes both **Gradient Descent** and **Normal Equation** solvers,built-in evaluation metrics, and data visualization.
---
## ✨ Key Features
- **Dual Solvers**: `fit()` your model using iterative **Gradient Descent** or the closed-form **Normal Equation**.
- **Accurate Predictions**: Generate continuous predictions with `predict()`.
- **Built-in Metrics**: Easily evaluate performance using **MSE, RMSE, MAE, and R² Score**.
- **Robust Implementation**: Includes automatic feature scaling (standardization) and bias term (intercept) support.
- **Clean Architecture**: Designed as a highly readable, reusable Python class.
---
## 🧠 The Math Behind the Model
This implementation relies on the core mathematical foundations of Machine Learning:
- **Hypothesis Function**:  
  $$ \hat{y} = X \theta $$
- **Cost Function (Mean Squared Error)**:  
  $$ J(\theta) = \frac{1}{2m} \| X \theta - y \|^2 $$
- **Gradient Descent Update Rule**:  
  $$ \theta \leftarrow \theta - \eta \nabla_\theta J(\theta) $$
- **Normal Equation (Closed Form)**:  
  $$ \theta = (X^T X)^{-1} X^T y $$
---
## 🗂️ Project Structure
```text
├── linear-regressionN.ipynb   # Jupyter Notebook with full implementation & plots
├── README.md                  # Project Documentation
└── requirements.txt           # Python dependencies
🚀 Quick Start / Usage
python


import numpy as np
from linear_regression import LinearRegression
# 1. Create a dummy dataset (y ≈ 2x + 1)
X = np.array([[1], [2], [3], [4], [5]], dtype=float)
y = np.array([3, 5, 7, 9, 11], dtype=float)  
# 2. Initialize the model 
# (You can also use method="normal" for the Normal Equation)
model = LinearRegression(method="gd", lr=0.05, epochs=2000)
# 3. Train the model
model.fit(X, y)
# 4. Evaluate & Predict
print(f"Learned Parameters (Theta): {model.theta}")
print(f"R² Score: {model.r2_score(X, y):.4f}")
print(f"Prediction for x=6: {model.predict(np.array([[6.0]]))[0]:.2f}")
📦 Installation
To run this project locally, clone the repository and install the required dependencies:

bash


# Clone the repository
git clone https://github.com/Bisma474/custom-linear-regression.git
cd custom-linear-regression
# Install dependencies
pip install -r requirements.txt
requirements.txt

text


numpy>=1.21.0
matplotlib>=3.4.0
pandas>=1.3.0
✍️ Author
👩‍💻 Bisma Munir
