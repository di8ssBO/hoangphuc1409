import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pickle
from utils.visualization import *

data = pd.read_csv('Humidity_Temp_Prediction.csv', sep = ";", parse_dates= ["date_time"])
data["minute"] = data["date_time"].dt.minute # Minute in Hours
data["hour"] = data["date_time"].dt.hour
data["minute"] = data["minute"] + data["hour"]*60 # Minute in Days

feature_X = "minute"
feature_y = "temp"
data_group = data[[feature_X, feature_y]].groupby(feature_X, as_index = False).mean()
print(data_group.head(10))

X = data_group[[feature_X]].values 
y = data_group[feature_y].values 

print(X)
print(y)

filename = 'model/poly_reg.pkl'
poly_reg = pickle.load(open(filename, 'rb'))

filename = 'model/finalized_model.pkl'
model = pickle.load(open(filename, 'rb'))

y_predict = model.predict(poly_reg.fit_transform(X))
print(y)
print(y_predict)

SensorVizWithPrediction(data_group, feature_X, feature_y, y_predict)

residuals = y - y_predict 
print(residuals)
plt.figure(figsize = (20,5))
sns.displot(residuals)
plt.axvline(x = np.mean(residuals), color = 'red', label = 'mean')
plt.axvline(x = np.median(residuals), color = 'orange', label = 'median')
plt.xlabel("Residuals")
plt.legend(loc = "upper right")
plt.savefig(f'outputs/Residuals {feature_X} Vs {feature_y}.jpg')
plt.show()

print(np.mean(residuals) - 3*np.std(residuals))
print(np.mean(residuals) + 3*np.std(residuals))

std = np.std(residuals) 
y_predict_upBound = y_predict + 3*std
y_predict_lowBound = y_predict - 3*std

plt.figure(figsize=(18, 5))
plt.scatter(X,y, color = "blue")
plt.plot(X, y_predict, color = "red", linewidth = 5)
plt.plot(X, y_predict_upBound, color = "green", linewidth = 1)
plt.plot(X, y_predict_lowBound, color = "green", linewidth = 1)
plt.title(f"{feature_X} vs {feature_y}")
plt.xlabel(feature_X, size = 15)
plt.ylabel(feature_y, size = 15)
plt.savefig(f'outputs/Warnings {feature_X} Vs {feature_y}.jpg')
plt.show()

LogDF = pd.DataFrame(columns = ["minute", "true_temp", "predicted_temp", "conf_lower", "conf_upper"])
LogDF["minute"] = X.reshape(-1)
LogDF["true_temp"] = y
LogDF["predicted_temp"] = y_predict
LogDF["conf_lower"] = y_predict_lowBound
LogDF["conf_upper"] = y_predict_upBound
LogDF["Alarm"] = (LogDF["true_temp"] < LogDF["conf_lower"]) | (LogDF["true_temp"] > LogDF["conf_upper"])
print(LogDF.head())
print("Alarm Time : ")
print(LogDF[LogDF["Alarm"] == True])

