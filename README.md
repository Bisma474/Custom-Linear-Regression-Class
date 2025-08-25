# 📘 Custom Linear Regression (from Scratch)

A from-scratch implementation of **Linear Regression** using only core Python/NumPy.  
This project demonstrates the math and implementation behind regression, without relying on scikit-learn.  
It includes **gradient descent**, **normal equation**, evaluation metrics, and visualization.

---

## ✨ Features
- `fit()` using **Gradient Descent** and **Normal Equation**
- Predictions with `predict()`  
- Metrics: **MSE, RMSE, MAE, R²**  
- Feature scaling & bias term support  
- Clear, reusable class structure

---

## 🧠 Math (Brief Overview)
- Hypothesis: \( \hat{y} = X \theta \)  
- Loss: \( J(\theta) = \frac{1}{2m} \| X \theta - y \|^2 \)  
- Gradient Descent Update: \( \theta \leftarrow \theta - \eta \nabla_\theta J(\theta) \)

---

## 🗂️ Project Structure
```
├── linear-regressionN.ipynb   # Jupyter Notebook implementation
├── README.md                  # Documentation
└── requirements.txt           # Python dependencies
```

---

## 🚀 Usage
```python
import numpy as np
from linear_regression import LinearRegression

# Dummy dataset
X = np.array([[1],[2],[3],[4],[5]], dtype=float)
y = np.array([3,5,7,9,11], dtype=float)  # y ≈ 2x + 1

# Initialize model
model = LinearRegression(method="gd", lr=0.05, epochs=2000)

# Train model
model.fit(X, y)

# Predictions
print("Theta:", model.theta)
print("R2 Score:", model.r2_score(X, y))
print("Prediction for x=6:", model.predict(np.array([[6.0]])))
```

---

## 📦 Installation
```bash
pip install -r requirements.txt
```

**requirements.txt**
```
numpy
matplotlib
pandas
```

---

---

## ✍️ Author
👩‍💻 **Bisma Munir**

