import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# Data
years = np.array([2024, 2025, 2026, 2027, 2028, 2029])
y = np.array([691, 599, 534, 478, 430, 388])

# Define exponential function: y = a * exp(b*x)
def exp_func(x, a, b):
    return a * np.exp(b * (x - years[0]))  # shift to avoid huge exponents

# Fit curve
params, _ = curve_fit(exp_func, years, y, p0=(700, -0.1))
a, b = params

# Predicted values
y_pred = exp_func(years, a, b)

# R² score
r2 = r2_score(y, y_pred)

print(r2_score)