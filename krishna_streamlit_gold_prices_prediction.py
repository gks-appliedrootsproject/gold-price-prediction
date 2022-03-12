import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import date,datetime

import random
import math
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import *

from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score

import numpy as np # for some math operations
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

# remember that our testing data set is only for the year 2018
# and we will use year below 2018 for training
features = df.drop(columns=['Close','Adj Close'],axis=1)
target = df['Adj Close']

year_2018_index_start = 1470

X_train = features[:year_2018_index_start]
X_test = features[year_2018_index_start:]

y_train = target[:year_2018_index_start]
y_test = target[year_2018_index_start:]


value_to_use_selection = st.radio("What would you like to do? ", ("None","Enter my own data points", 'Let the system automatically select a random data point from test set','I will pick a date'))
# If you want to use only 3 features and enter your own values then, please add this to above -->  ,"Enter my own data points"

i = 0
# uncomment the below 3 lines if you want to use only 3 features
custom_open = 0
custom_low = 0
custom_high = 0

random_date = []
value_to_predict = 0
lr_fit = {}
dectree_fit = {}
rf_fit = {}

if value_to_use_selection == "Let the system automatically select a random data point from test set":
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

elif value_to_use_selection == "I will pick a date":

    st.markdown(" - *Please select a business/work day, **use dates only from the year 2018** and no weekends or Government holidays*" )
    st.markdown(" - *After picking your custom date,* **if you get any error, then please select a valid date** ")
   
    pick_date = st.date_input( " Pick a date from the test dataset (only from the year 2018 & no weekends or Government holidays)",min_value=date(2018, 1, 2), max_value=date(2018,12,31),value=date(2018,1,2),help=" ## No weekends or Government holidays")
    st.write(pick_date)
    
    if pd.to_datetime(pick_date)  in X_test.index :
        custom_date =  X_test.loc[[pick_date]]
        st.write(f"###### You have picked this date",pick_date)
        st.write(" Here is the test data point(*independent variables/features*) on your picked date",custom_date)
        value_to_predict = custom_date
        lr_fit = joblib.load("lr_full.pkl")
        dectree_fit = joblib.load("dectree_full.pkl")
        rf_fit = joblib.load("rf_full.pkl")             
     
    else:
        st.warning(" ##### Error: This is not  a valid test date, so please pick any other date (no weekends or government holidays)")
        
    
# uncomment below 10 lines if you want to use only 3 features
elif value_to_use_selection == "Enter my own data points":
    custom_open = st.number_input("Insert a value for: Open",min_value = 100.0,step=5.0)
    custom_low = st.number_input("Insert a value for: Low",min_value = 100.0,step=5.0)
    custom_high = st.number_input("Insert a value for: High",min_value = 100.0,step=5.0)
    X_train = X_train[["Open", "Low", "High"]]
    X_test = X_test[["Open", "Low", "High"]]
    value_to_predict = pd.DataFrame([[custom_open, custom_low, custom_high]], columns=["Open", "Low", "High"])
    lr_fit = joblib.load("lr_small.pkl")
    dectree_fit = joblib.load("dectree_small.pkl")
    rf_fit = joblib.load("rf_small.pkl")

elif  value_to_use_selection != "None":
    st.write("")

else:
    st.write("Please make a selection for what value to test with.")

Algorithm = {}
if value_to_use_selection != "None" :

    Algorithm = st.selectbox( "Pick a Machine Learning  Algorithm:",["None","Linear Regression", "Decision Tree Regression","Random Forest Regression"])

    prediction = 0
    y_pred = []

    if Algorithm == "Linear Regression" :
        st.markdown("<h4 style='text-align: center; color: blue;'>Linear Regression  </h4>", unsafe_allow_html=True)
        prediction = lr_fit.predict(value_to_predict)[0]
        y_pred = lr_fit.predict(X_test)

    elif Algorithm == "Decision Tree Regression":
        st.markdown("<h4 style='text-align: center; color: blue;'>Decision Tree Regression</h4>", unsafe_allow_html=True)
        prediction = dectree_fit.predict(value_to_predict)[0]
        y_pred = dectree_fit.predict(X_test)

    elif Algorithm == "Random Forest Regression":
        st.markdown("<h4 style='text-align: center; color: blue;'>Random Forest Regression</h4>", unsafe_allow_html=True)
        prediction = rf_fit.predict(value_to_predict)[0]
        y_pred = rf_fit.predict(X_test)

    
    else:
        st.write("#### Pick a Machine Learning Algorithm from the above drop down menu.")


    if Algorithm != "None" and value_to_use_selection != "I will pick a date" :
        st.write(f"##### We can make predictions on this data ", value_to_predict)
        if st.button("Predict"):
            # Output prediction
            st.write(f"### `The predicted Adj Close value is:` {prediction}")
            
            if value_to_use_selection != "Enter my own data points":
                st.write(f"### `The actual Adj Close value on` ",random_date.index[0],"is:",y_test.loc[[random_date.index[0]]])
                my_custom_df1 = pd.DataFrame({'Actual': y_test.loc[[random_date.index[0]]], 'Predicted': prediction})
                
                st.markdown("<h4 style='text-align: center; color: blue;'> Actual  vs. Predicted</h4>", unsafe_allow_html=True)
                fig2 =px.bar(my_custom_df1,barmode='group')
                fig2.update_xaxes(tickformat="%b %d\n%Y")
                fig2.update_layout(xaxis_title = "Your Selected Date",yaxis_title = "Adj Close")
                st.plotly_chart(fig2)
            

            
            
            # If you want to show metrics then please uncomment the below code block
            #st.markdown("<h2 style='text-align: center; color: blue;'> Metrics for the model comparing training data set vs. testing data set:</h2>", unsafe_allow_html=True)
            #r2_score_lr = r2_score(y_test,y_pred)
            #rmse_lr = math.sqrt(mean_squared_error( y_true =y_test,y_pred= y_pred))
            #explained_variance = explained_variance_score(y_test, y_pred)

            #code block for above metrics
            #col1,col2,col3 = st.columns(3)
            #col1.metric("The r2 score is",r2_score_lr)
            #col2.metric("RMSE is",rmse_lr)
            #col3.metric("Explained variance is",explained_variance)
            #st.markdown("The best possible explained variance score  is 1.0, lower values are worse.\n"
            #           " If this value is closer to 1 ,then it indicates a stronger strength of association.\n"
            #           "It also means that we  make better predictions")

                
    

    elif  Algorithm == "None" and value_to_use_selection == "Let the system automatically select a random data point from test set":
        st.write("")
    
    elif  Algorithm == "None" and value_to_use_selection == "Enter my own data points":
        st.write("")

    elif Algorithm == "None" and value_to_use_selection == "I will pick a date":
        st.write("")
    
    else:
        st.write(f"##### We will use below selected data point to make prediction: ", value_to_predict)
        if st.button("Predict"):
            # Output prediction
            st.write(f" ### `The predicted Adj Close value is:` {prediction}")

            st.write(f"### `The actual Adj Close value on` ",pick_date,"is:",y_test.loc[[pick_date]])
            my_custom_df = pd.DataFrame({'Actual': y_test.loc[[pick_date]], 'Predicted': prediction})

            st.markdown("<h4 style='text-align: center; color: blue;'> Actual  vs. Predicted</h4>", unsafe_allow_html=True)
            fig2 =px.bar(my_custom_df,barmode='group')
            fig2.update_xaxes(tickformat="%b %d\n%Y")
            fig2.update_layout(xaxis_title = "Your Selected Date",yaxis_title = "Adj Close")
            st.plotly_chart(fig2)


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
    fig.update_layout(yaxis_title = "Adj Close")
    st.plotly_chart(fig)

st.sidebar.caption("# About Data:",unsafe_allow_html=True)
with st.sidebar.expander('Please click here (or use + sign) to get data details'):
    st.markdown("""
Gold ETFs (exchange-traded-funds) can be purchased like shares on a stock exchange.\n

Instead  of holding physical gold,the investors own small quantities of gold-related assets that consists of only Gold as principal asset.\n

**The data has `1718 rows and 80 columns.`**\n

**Features/independent variables are:**\n
    Gold ETF :- Date, Open, High, Low, Close and Volume.
    S&P 500 Index :- 'SP_open', 'SP_high', 'SP_low', 'SP_close', 'SP_Ajclose', 'SP_volume'
    Dow Jones Index :- 'DJ_open','DJ_high', 'DJ_low', 'DJ_close', 'DJ_Ajclose', 'DJ_volume'
    Eldorado Gold Corporation (EGO) :- 'EG_open', 'EG_high', 'EG_low', 'EG_close', 'EG_Ajclose', 'EG_volume'
    EURO - USD Exchange Rate :- 'EU_Price','EU_open', 'EU_high', 'EU_low', 'EU_Trend'
    Brent Crude Oil Futures :- 'OF_Price', 'OF_Open', 'OF_High', 'OF_Low', 'OF_Volume', 'OF_Trend'
    Crude Oil WTI USD :- 'OS_Price', 'OS_Open', 'OS_High', 'OS_Low', 'OS_Trend'
    Silver Futures :- 'SF_Price', 'SF_Open', 'SF_High', 'SF_Low', 'SF_Volume', 'SF_Trend'
    US Bond Rate (10 years) :- 'USB_Price', 'USB_Open', 'USB_High','USB_Low', 'USB_Trend'
    Platinum Price :- 'PLT_Price', 'PLT_Open', 'PLT_High', 'PLT_Low','PLT_Trend'
    Palladium Price :- 'PLD_Price', 'PLD_Open', 'PLD_High', 'PLD_Low','PLD_Trend'
    Rhodium Prices :- 'RHO_PRICE'
    US Dollar Index : 'USDI_Price', 'USDI_Open', 'USDI_High','USDI_Low', 'USDI_Volume', 'USDI_Trend'
    Gold Miners ETF :- 'GDX_Open', 'GDX_High', 'GDX_Low', 'GDX_Close', 'GDX_Adj Close', 'GDX_Volume'
    Oil ETF USO :- 'USO_Open','USO_High', 'USO_Low', 'USO_Close', 'USO_Adj Close', 'USO_Volume'
**Target Variable is:** `Adjusted Close` """,unsafe_allow_html=True)
