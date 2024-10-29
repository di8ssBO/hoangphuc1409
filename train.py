import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pickle
from utils.visualization import *

print("Hello World")

data = pd.read_csv('Humidity_Temp_Prediction.csv', sep = ";", parse_dates= ["date_time"])
print(data.shape)
print(data.info())
print(data.describe())

data["minute"] = data["date_time"].dt.minute # Minute in Hours
data["hour"] = data["date_time"].dt.hour
data["minute"] = data["minute"] + data["hour"]*60 # Minute in Days
print(data.head())

feature_y = "temp"
feature_X = "minute"
data_group = data[[feature_X, feature_y]].groupby(feature_X, as_index = False).mean()
SensorViz(data_group, feature_X, feature_y)


feature_X = "minute"
feature_y = "temp"
data_group = data[[feature_X, feature_y]].groupby(feature_X, as_index = False).mean()
print(data_group.head(10))

X = data_group[[feature_X]].values 
y = data_group[feature_y].values 

print(X)
print(y)

poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
print(X_poly[:5])

model = LinearRegression()
model.fit(X_poly, y)

print("Incorrect Input :", X_poly[0]) 
print("Correct Input :", X_poly[0].reshape(1, -1)) 
print("Output : ", model.predict(X_poly[0].reshape(1, -1)))

filename = 'model/poly_reg.pkl'
pickle.dump(poly_reg, open(filename, 'wb'))

filename = 'model/finalized_model.pkl'
pickle.dump(model, open(filename, 'wb'))
