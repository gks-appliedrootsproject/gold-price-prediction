import streamlit as st

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

import joblib

# Title

st.header("Applied Roots: Gold Prices ML Prediction App")
st.subheader("by Dr. Ganti Krishna Sarma")

@st.cache(persist=True)
def fetch_data():
    data = pd.read_csv("FINAL_USO.csv",parse_dates=['Date'],index_col=['Date'])
    return data
df = fetch_data()

st.write(f" ##### Top 3 rows of the dataset")
st.dataframe(df.head(3))

st.write(f" ##### Bottom 3 rows of the dataset")
st.dataframe(df.tail(3))

# Sidebar
st.sidebar.markdown("#### Please make your selections from below options")
if st.sidebar.checkbox("Show independent variables data", False):
    st.subheader("Independent variables interactive dataframe")
    st.dataframe(features)

if st.sidebar.checkbox("Show Line plots of Independent variables", False):
    st.subheader("Independent variables line plots")
    fig = px.line(features)
    st.plotly_chart(fig)

if st.sidebar.checkbox(" Show Line plot of  the target variable(Adj Close)", False):
    st.subheader("Adj Close line plot")
    fig = px.line(target)
    st.plotly_chart(fig)

# remember that our testing data set is only for the year 2018
# and we will use year below 2018 for training
features = df.drop(columns=['Close','Adj Close'],axis=1)
target = df['Adj Close']

year_2018_index_start = 1470
X_train = features[:year_2018_index_start]
X_test = features[year_2018_index_start:]

y_train = target[:year_2018_index_start]
y_test = target[year_2018_index_start:]

value_to_use_selection = st.radio("What would you like to do? ", ("None", 'Pick a random data point from test set', 'Enter my own data points'))
i = 0
custom_open = 0
custom_low = 0
custom_high = 0
random_date = []
value_to_predict = 0
lr_fit = {}
dectree_fit = {}
rf_fit = {}
if value_to_use_selection == "Pick a random data point from test set":
    # Pick a random date from the test set
    i = random.choice(range(len(X_test)))
    random_date = X_test.iloc[[i]]
    n_feats = len(random_date.columns)
    st.write(f" ##### Here is a randomly selected date from the test set: ", random_date)
    st.write(f" ###### From the above",n_feats,"features, let's predict Adj Close value")
    value_to_predict = random_date
    lr_fit = joblib.load("lr_full.pkl")
    dectree_fit = joblib.load("dectree_full.pkl")
    rf_fit = joblib.load("rf_full.pkl")
elif value_to_use_selection == "Enter my own data points":
    custom_open = st.number_input("Insert a value for: Open")
    custom_low = st.number_input("Insert a value for: Low")
    custom_high = st.number_input("Insert a value for: High")
    X_train = X_train[["Open", "Low", "High"]]
    X_test = X_test[["Open", "Low", "High"]]
    value_to_predict = pd.DataFrame([[custom_open, custom_low, custom_high]], columns=["Open", "Low", "High"])
    lr_fit = joblib.load("lr_small.pkl")
    dectree_fit = joblib.load("dectree_small.pkl")
    rf_fit = joblib.load("rf_small.pkl")
else:
    st.write("Please make a selection for what value to test with.")

classifier = {}
if value_to_use_selection != "None":
    classifier = st.selectbox( "Make a selection for the classifier:",["None","Linear Regression", "Decision Tree","Random Forest"])

    prediction = 0
    y_pred = []

    if classifier == "Linear Regression":
        st.markdown("<h4 style='text-align: center; color: blue;'>Linear Regression  </h4>", unsafe_allow_html=True)
        prediction = lr_fit.predict(value_to_predict)[0]
        y_pred = lr_fit.predict(X_test)

    elif classifier == "Decision Tree":
        st.markdown("<h4 style='text-align: center; color: blue;'>Decision Tree</h4>", unsafe_allow_html=True)
        prediction = dectree_fit.predict(value_to_predict)[0]
        y_pred = dectree_fit.predict(X_test)

    elif classifier == "Random Forest":
        st.markdown("<h4 style='text-align: center; color: blue;'>Random Forest</h4>", unsafe_allow_html=True)
        prediction = rf_fit.predict(value_to_predict)[0]
        y_pred = rf_fit.predict(X_test)

    else:
        st.write("#### Pick a classifier from the drop down menu.")


    if classifier != "None":
        st.write(f"##### Value to be predicted is: ", value_to_predict)
        if st.button("Predict"):
            # Output prediction
            st.write(f"#### The predicted Adj Close value is: {prediction}")
            my_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

            # st.markdown("<h4 style='text-align: center; color: blue;'> train vs. test data</h4>", unsafe_allow_html=True)
            # fig2 =px.line(my_df)
            # st.plotly_chart(fig2)

            st.markdown("<h2 style='text-align: center; color: blue;'> Metrics for the model comparing training data set vs. testing data set:</h2>", unsafe_allow_html=True)
            r2_score_lr = r2_score(y_test,y_pred)
            rmse_lr = math.sqrt(mean_squared_error( y_true =y_test,y_pred= y_pred))
            explained_variance = explained_variance_score(y_test, y_pred)

            #code block for above metrics
            col1,col2,col3 = st.columns(3)
            col1.metric("The r2 score is",r2_score_lr)
            col2.metric("RMSE is",rmse_lr)
            col3.metric("Explained variance is",explained_variance)
            st.markdown("The best possible explained variance score  is 1.0, lower values are worse.\n"
                       " If this value is closer to 1 ,then it indicates a stronger strength of association.\n"
                        "It also means that we  make better predictions")

