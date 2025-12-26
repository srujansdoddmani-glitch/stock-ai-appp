import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib

model = joblib.load("model.pkl")

st.title("Indian Stock AI Predictor 🇮🇳")

stock = st.text_input("Enter Stock Symbol","RELIANCE.NS")

if st.button("Predict"):
    data = yf.download(stock, period="2y")
    data["SMA_10"] = data["Close"].rolling(10).mean()
    data["SMA_50"] = data["Close"].rolling(50).mean()
    latest = data.dropna().iloc[-1]
    
    X = [[latest["Close"], latest["SMA_10"], latest["SMA_50"]]]
    pred = model.predict(X)[0]

    if pred == 1:
        st.success("Prediction: Price May Go UP Tomorrow 📈")
    else:
        st.error("Prediction: Price May Go DOWN Tomorrow 📉")
