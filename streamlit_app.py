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

# Generate random sentiment data (simulation)
def generate_sentiment_data(dates):
    np.random.seed(42)  # For reproducibility
    sentiment_values = np.random.normal(loc=0.2, scale=0.3, size=len(dates))
    return pd.Series(sentiment_values, index=dates)

if st.button("Run Backtest"):
    with st.spinner("Fetching data..."):
        # Fetch NVIDIA stock data
        ticker = yf.Ticker("NVDA")
        df = ticker.history(start=start_date, end=end_date)
        
        if not df.empty:
            # Generate simulated sentiment data
            sentiment = generate_sentiment_data(df.index)
            
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
                title={
                    'text': 'NVIDIA Stock Performance & Market Sentiment',
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'size': 24, 'color': '#006400'}
                },
                yaxis=dict(
                    title='Stock Price (USD)',
                    titlefont=dict(color='#76b900'),
                    tickfont=dict(color='#76b900')
                ),
                yaxis2=dict(
                    title='Sentiment Score',
                    titlefont=dict(color='#006400'),
                    tickfont=dict(color='#006400'),
                    overlaying='y',
                    side='right'
                ),
                height=700,
                template='plotly_white',
                hovermode='x unified',
                plot_bgcolor='white',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor='rgba(255, 255, 255, 0.8)'
                )
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
            
            # Calculate and display correlation
            correlation = df['Close'].corr(sentiment)
            st.write(f"Price-Sentiment Correlation: {correlation:.2f}")
        else:
            st.error("No data available for the selected date range.")

# Add note about sentiment
st.sidebar.info("Note: This is a demonstration using historical stock data from Yahoo Finance with simulated sentiment scores.")
