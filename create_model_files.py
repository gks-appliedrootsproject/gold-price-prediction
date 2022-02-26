import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

import random
import math
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import *

from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score

import numpy as np # for som math operations
from sklearn.preprocessing import StandardScaler # for standardizing the Data

from statsmodels.tsa.arima.model import ARIMA,ARIMAResults
import statsmodels.api as smapi

import joblib

def fetch_data():
    data = pd.read_csv("FINAL_USO.csv",parse_dates=['Date'],index_col=['Date'])
    return data
df = fetch_data()

# define features
features = df.drop(columns=['Close','Adj Close'],axis=1)
# define target
target = df['Adj Close']

# remember that our testing data set is only for the year 2018
# and we will use year below 2018 for training
year_2018_index_start = 1470

X_train = features[:year_2018_index_start]
X_test = features[year_2018_index_start:]

y_train = target[:year_2018_index_start]
y_test = target[year_2018_index_start:]

X_train_small = X_train[["Open", "Low", "High"]]
X_test_small = X_test[["Open", "Low", "High"]]

lr = LinearRegression()
joblib.dump(lr.fit(X_train, y_train), "lr_full.pkl")
joblib.dump(lr.fit(X_train_small, y_train), "lr_small.pkl")

dectree = DecisionTreeRegressor(random_state=2022) # picking a random value for the random_state
joblib.dump(dectree.fit(X_train, y_train), "dectree_full.pkl")
joblib.dump(dectree.fit(X_train_small, y_train), "dectree_small.pkl")

rf = RandomForestRegressor(n_estimators=10)
joblib.dump(rf.fit(X_train, y_train), "rf_full.pkl")
joblib.dump(rf.fit(X_train_small, y_train), "rf_small.pkl")
