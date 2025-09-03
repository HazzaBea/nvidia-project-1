import streamlit as st
import requests
from datetime import datetime

st.title("NVIDIA Market Sentiment")

FINNHUB_API_KEY = 'd2s0nrpr01qv11lgk070d2s0nrpr01qv11lgk07g'

# Fetch stock price
try:
    response = requests.get(f"https://finnhub.io/api/v1/quote?symbol=NVDA&token={FINNHUB_API_KEY}")
    data = response.json()
    price = data.get('c')
    if price:
        st.metric("NVDA Stock Price", f"${price:.2f}")
except Exception as e:
    st.error(f"Error fetching stock price: {str(e)}")

# Simple sentiment display
st.subheader("Market Sentiment")
sentiment = 0.5  # Placeholder sentiment
st.progress(sentiment)
st.write("Sentiment: Neutral")
