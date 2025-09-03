import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

# Page config
st.set_page_config(
    page_title="NVIDIA Market Sentiment Backtesting v2",  # Added version number
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

# Set title with custom color
st.markdown("<h1 style='color: #006400;'>NVIDIA Market Sentiment Backtesting</h1>", unsafe_allow_html=True)

def get_historical_data(start_date, end_date):
    """Fetch historical stock data from Yahoo Finance"""
    try:
        # Create a Ticker object for NVIDIA
        nvda = yf.Ticker("NVDA")
        
        # Fetch the historical data
        df = nvda.history(
            start=start_date,
            end=end_date,
            interval="1d"
        )
        
        if not df.empty:
            return df
        else:
            st.error("No data received from Yahoo Finance")
            return None
    except Exception as e:
        st.error(f"Error fetching historical data: {str(e)}")
        return None

def get_historical_news(start_date, end_date):
    """Fetch historical news and calculate sentiment"""
    try:
        url = f"https://finnhub.io/api/v1/company-news?symbol=NVDA&from={start_date}&to={end_date}&token={FINNHUB_API_KEY}"
        response = requests.get(url)
        news = response.json()
        
        # Process news and calculate daily sentiment
        daily_sentiment = {}
        
        for article in news:
            date = datetime.fromtimestamp(article['datetime']).strftime('%Y-%m-%d')
            # Simulate sentiment (replace with actual sentiment analysis)
            import random
            sentiment = random.uniform(-1, 1)
            
            if date in daily_sentiment:
                daily_sentiment[date].append(sentiment)
            else:
                daily_sentiment[date] = [sentiment]
        
        # Calculate average daily sentiment
        sentiment_df = pd.DataFrame({
            'Date': daily_sentiment.keys(),
            'Sentiment': [sum(scores)/len(scores) for scores in daily_sentiment.values()]
        })
        sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
        sentiment_df.set_index('Date', inplace=True)
        return sentiment_df
    except Exception as e:
        st.error(f"Error fetching news data: {str(e)}")
        return None

# UI
# Date inputs
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", 
                              value=datetime.now() - timedelta(days=365),  # 1 year
                              min_value=datetime.now() - timedelta(days=365),
                              max_value=datetime.now())
with col2:
    end_date = st.date_input("End Date",
                            value=datetime.now(),
                            min_value=start_date,
                            max_value=datetime.now())

if st.button("Run Backtest"):
    with st.spinner("Running backtest..."):
        # Convert dates to string format
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        # Fetch data
        price_data = get_historical_data(start_str, end_str)
        sentiment_data = get_historical_news(start_str, end_str)
        
        if price_data is not None and sentiment_data is not None:
            # Create subplot
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add price line with area fill
            fig.add_trace(
                go.Scatter(
                    x=price_data.index, 
                    y=price_data['Close'],
                    name="NVIDIA Stock Price", 
                    line=dict(color="#76b900", width=2),  # NVIDIA's brand green
                    fill='tonexty',
                    fillcolor='rgba(118, 185, 0, 0.1)'  # Light green fill
                ),
                secondary_y=False
            )
            
            # Add sentiment line
            fig.add_trace(
                go.Scatter(
                    x=sentiment_data.index, 
                    y=sentiment_data['Sentiment'],
                    name="Market Sentiment", 
                    line=dict(color="#006400", width=2, dash='dot')
                ),
                secondary_y=True
            )
            
            # Update layout
            fig.update_layout(
                title={
                    'text': "NVIDIA Stock Price vs Market Sentiment (1-Year View)",
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'size': 24, 'color': '#006400'}
                },
                xaxis_title="Date",
                plot_bgcolor='white',
                height=700,
                hovermode='x unified',
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
            
            # Update y-axes labels
            fig.update_yaxes(title_text="Stock Price ($)", secondary_y=False)
            fig.update_yaxes(title_text="Sentiment Score", secondary_y=True)
            
            # Display plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate statistics
            price_change = ((price_data['Close'][-1] - price_data['Close'][0]) / price_data['Close'][0]) * 100
            avg_sentiment = sentiment_data['Sentiment'].mean()
            
            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Price Change", f"{price_change:.2f}%")
            with col2:
                st.metric("Average Sentiment", f"{avg_sentiment:.2f}")
            
            # Display correlation
            correlation = price_data['Close'].corr(sentiment_data['Sentiment'])
            st.write(f"Price-Sentiment Correlation: {correlation:.2f}")

# Add a note about the sentiment calculation
st.sidebar.info("Note: This is a demonstration using simulated sentiment scores. In a production environment, you would want to use a more sophisticated sentiment analysis model.")
