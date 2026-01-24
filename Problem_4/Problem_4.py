# 4.	Problem Statement

# Write a Python program to draw a 3D plot that visualizes the regression model for house price prediction using suitable Python-based 3D plotting libraries.

# Assume the following features were used:
# •	Area (sq ft)
# •	Number of Bedrooms
# •	House Price

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
 
area = np.array([800, 1000, 1200, 1500, 1800, 2000, 2300, 2600, 3000])
 
bedrooms = np.array([1, 2, 2, 3, 3, 4, 4, 5, 5])
 
price = np.array([40, 50, 55, 70, 85, 95, 120, 150, 180])
 
X = np.column_stack((area, bedrooms))
y = price
 
model = LinearRegression()
model.fit(X, y)
 
area_range = np.linspace(area.min(), area.max(), 30)
bedroom_range = np.linspace(bedrooms.min(), bedrooms.max(), 30)
area_grid, bedroom_grid = np.meshgrid(area_range, bedroom_range)
 
grid_points = np.column_stack((area_grid.ravel(), bedroom_grid.ravel()))
price_pred = model.predict(grid_points)
price_grid = price_pred.reshape(area_grid.shape)
 
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
 
ax.scatter(area, bedrooms, price, marker='o')
 
ax.plot_surface(area_grid, bedroom_grid, price_grid, alpha=0.5)
 
ax.set_xlabel("Area (sq ft)")
ax.set_ylabel("Number of Bedrooms")
ax.set_zlabel("House Price (in Lakhs)")
ax.set_title("3D Visualization of House Price Regression Model")
 
plt.show()
