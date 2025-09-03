import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import numpy as np

# Page config
st.set_page_config(
    page_title="NVIDIA Market Sentiment Backtesting",
    layout="wide"
)

# Custom CSS for dark green text
st.markdown("""
    <style>
        div[data-testid="stText"],
        div[data-testid="stMarkdown"],
        div[data-testid="stHeader"],
        div[data-baseweb="select"] > div,
        div[data-testid="stMetricValue"],
        div[data-testid="stMetricLabel"],
        .streamlit-expanderHeader,
        .stTitle,
        p,
        h1, h2, h3, h4, h5, h6 {
            color: #006400 !important;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='color: #006400;'>NVIDIA Market Sentiment Backtesting</h1>", unsafe_allow_html=True)

# Date inputs
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", 
                              value=datetime.now() - timedelta(days=365),
                              min_value=datetime.now() - timedelta(days=365),
                              max_value=datetime.now())
with col2:
    end_date = st.date_input("End Date", 
                            value=datetime.now(),
                            min_value=start_date,
                            max_value=datetime.now())

# Fetch and process news sentiment data
def get_news_sentiment(start_date, end_date):
    # Alpha Vantage API endpoint
    ALPHA_VANTAGE_API_KEY = st.secrets["ALPHA_VANTAGE_API_KEY"]
    base_url = "https://www.alphavantage.co/query"
    
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": "NVDA",
        "time_from": start_date.strftime("%Y%m%dT0000"),
        "time_to": end_date.strftime("%Y%m%dT2359"),
        "limit": 1000,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        
        if "feed" not in data:
            st.error(f"API Error: {data.get('Note', 'Unknown error')}")
            return None
            
        # Process news articles
        sentiment_data = []
        for article in data["feed"]:
            date = pd.to_datetime(article["time_published"]).strftime("%Y-%m-%d")
            sentiment_score = float(article["overall_sentiment_score"])
            sentiment_data.append({
                "date": date,
                "sentiment": sentiment_score
            })
        
        # Convert to DataFrame and calculate daily average sentiment
        if sentiment_data:
            df_sentiment = pd.DataFrame(sentiment_data)
            daily_sentiment = df_sentiment.groupby("date")["sentiment"].mean()
            daily_sentiment.index = pd.to_datetime(daily_sentiment.index)
            
            # Forward fill missing dates
            idx = pd.date_range(start_date, end_date)
            daily_sentiment = daily_sentiment.reindex(idx).fillna(method='ffill').fillna(0)
            
            return daily_sentiment
        else:
            st.warning("No news sentiment data available for the selected date range")
            return None
            
    except Exception as e:
        st.error(f"Error fetching news sentiment: {str(e)}")
        return None

if st.button("Run Backtest"):
    with st.spinner("Fetching data..."):
        # Fetch NVIDIA stock data
        ticker = yf.Ticker("NVDA")
        df = ticker.history(start=start_date, end=end_date)
        
        if not df.empty:
            # Get real news sentiment data
            sentiment = get_news_sentiment(start_date, end_date)
            
            if sentiment is None:
                st.error("Could not fetch sentiment data. Please try again later.")
                return
            
            # Create the plot with two y-axes
            fig = go.Figure()
            
            # Add stock price line
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Close'],
                name='NVDA Stock Price',
                line=dict(color='#76b900', width=2),  # NVIDIA green
                fill='tonexty',
                fillcolor='rgba(118,185,0,0.1)'
            ))
            
            # Add sentiment line on secondary y-axis
            fig.add_trace(go.Scatter(
                x=df.index,
                y=sentiment,
                name='Market Sentiment',
                line=dict(color='#006400', width=2, dash='dot'),
                yaxis='y2'
            ))
            
            # Update layout
            fig.update_layout(
                title='NVIDIA Stock Performance & Market Sentiment',
                yaxis_title='Stock Price (USD)',
                yaxis2=dict(
                    title='Sentiment Score',
                    overlaying='y',
                    side='right'
                ),
                height=700,
                showlegend=True
            )
            
            # Update axes
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate and display statistics
            price_change = ((df['Close'][-1] - df['Close'][0]) / df['Close'][0]) * 100
            avg_sentiment = sentiment.mean()
            
            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Price Change", f"{price_change:.2f}%")
            with col2:
                st.metric("Average Sentiment", f"{avg_sentiment:.2f}")
            
            # Calculate next day returns
            df['Next_Day_Return'] = df['Close'].pct_change().shift(-1)
            
            # Calculate correlations
            same_day_corr = df['Close'].corr(sentiment)
            next_day_corr = df['Next_Day_Return'].corr(sentiment)
            
            # Calculate prediction accuracy
            sentiment_signals = (sentiment > sentiment.mean()).astype(int)  # 1 if bullish, 0 if bearish
            actual_moves = (df['Next_Day_Return'] > 0).astype(int)  # 1 if price went up, 0 if down
            correct_predictions = (sentiment_signals == actual_moves).mean() * 100
            
            # Display analysis results
            st.markdown("### Sentiment Analysis Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Same-Day Correlation", f"{same_day_corr:.2f}")
            with col2:
                st.metric("Next-Day Correlation", f"{next_day_corr:.2f}")
            with col3:
                st.metric("Prediction Accuracy", f"{correct_predictions:.1f}%")
            
            # Add interpretation
            st.markdown("### Interpretation")
            st.markdown("""
            - **Same-Day Correlation**: Shows how sentiment aligns with current price movements
            - **Next-Day Correlation**: Shows how well sentiment predicts next day's price movement
            - **Prediction Accuracy**: Percentage of times sentiment correctly predicted price direction
            
            The sentiment score is calculated using real news articles about NVIDIA:
            - Articles are analyzed for sentiment using natural language processing
            - Multiple news sources are considered
            - Daily sentiment scores are averaged
            - Missing days use the last known sentiment (forward-fill)
            
            Note: This uses real market sentiment from news articles rather than technical indicators.
            """)
            
        else:
            st.error("No data available for the selected date range.")

# Add note about sentiment
st.sidebar.info("Note: This is a demonstration using historical stock data from Yahoo Finance with simulated sentiment scores.")
