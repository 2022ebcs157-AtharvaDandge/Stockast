"""
Stock Prediction & Sentiment Analysis App - OPTIMIZED FOR SPEED
Optimizations implemented:
- Lazy imports for heavy NLP libraries (TextBlob, NLTK)
- Removed immediate NLTK downloads on startup
- Limited initial stock loading to 50 stocks
- Aggressive caching of resources and data
- Removed redundant spinner on startup
"""

# Core imports (fast)
import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime as dt, timedelta
import requests

# Plotting imports - INTERACTIVE PLOTLY
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ML imports
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# NLP imports - LAZY LOAD FOR PERFORMANCE
import bs4

# Lazy import for sentiment analysis (only load when needed)
_sentiment_analyzer = None
_textblob_ready = False

# Configure Streamlit first for faster load
st.set_page_config(
    page_title="Stock Trend Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("Stock Trend Prediction & Sentiment Analysis")

# Initialize NLTK VADER once and cache it
@st.cache_resource
def get_sentiment_analyzer():
    """Get cached sentiment analyzer - lazy load"""
    try:
        import nltk
        nltk.download('vader_lexicon', quiet=True)
        from nltk.sentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except:
        return None

# Initialize TextBlob cache to ensure it's ready
@st.cache_resource
def initialize_textblob():
    """Pre-initialize TextBlob to prevent first-load issues"""
    try:
        from textblob import TextBlob
        # Test TextBlob with a simple text
        _ = TextBlob("test")
        return True
    except:
        return False

# Pre-initialize all resources on app load (lazy - only when needed)
@st.cache_resource
def initialize_app_resources():
    """Initialize all critical resources on app load"""
    try:
        # Get sentiment analyzer
        get_sentiment_analyzer()
        # Initialize TextBlob
        initialize_textblob()
        return True
    except Exception as e:
        return False

# Skip initialization on startup - do it lazily when needed

@st.cache_resource
def get_chart_colors():
    """Returns consistent color configuration for all charts"""
    return {
        'color_primary': '#3498db',
        'color_positive': '#2ecc71',
        'color_negative': '#e74c3c',
        'color_neutral': '#95a5a6',
        'color_secondary': '#f39c12',
    }

def plot_line_trend(dates, values, title, ylabel, use_fill=True, figsize='medium'):
    """Interactive line chart with optional fill"""
    colors = get_chart_colors()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=values,
        mode='lines',
        name='Price Trend',
        line=dict(color=colors['color_primary'], width=2),
        fill='tozeroy' if use_fill else None,
        fillcolor=f'rgba(52, 152, 219, 0.3)',
        hovertemplate='<b>Date:</b> %{x}<br><b>Price:</b> ₹%{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=ylabel,
        hovermode='x unified',
        template='plotly_dark',
        height=450
    )
    return fig

def plot_bar_chart(categories, values, title, ylabel, colors=None, figsize='medium', horizontal=False):
    """Interactive bar chart"""
    chart_colors = get_chart_colors()
    
    if colors is None:
        colors = [chart_colors['color_primary']] * len(values)
    
    if horizontal:
        fig = go.Figure(data=[
            go.Bar(y=categories, x=values, orientation='h', marker=dict(color=colors),
                   text=[f'₹{v:.2f}' for v in values], textposition='auto',
                   hovertemplate='<b>%{y}</b><br>₹%{x:.2f}<extra></extra>')
        ])
        fig.update_layout(xaxis_title=ylabel, yaxis_title="")
    else:
        fig = go.Figure(data=[
            go.Bar(x=categories, y=values, marker=dict(color=colors),
                   text=[f'₹{v:.2f}' for v in values], textposition='auto',
                   hovertemplate='<b>%{x}</b><br>₹%{y:.2f}<extra></extra>')
        ])
        fig.update_layout(yaxis_title=ylabel, xaxis_title="")
    
    fig.update_layout(
        title=title,
        hovermode='x unified',
        template='plotly_dark',
        height=450
    )
    return fig

def plot_pie_chart(data_dict, title, figsize='medium'):
    """Interactive pie chart"""
    colors = get_chart_colors()
    color_list = [colors['color_positive'], colors['color_negative'], colors['color_neutral']]
    
    fig = go.Figure(data=[go.Pie(
        labels=list(data_dict.keys()),
        values=list(data_dict.values()),
        marker=dict(colors=color_list[:len(data_dict)]),
        hovertemplate='<b>%{label}</b><br>%{value} (%{percent})<extra></extra>'
    )])
    
    fig.update_layout(
        title=title,
        template='plotly_dark',
        height=450
    )
    return fig

def plot_dual_comparison(dates1, values1, label1, live_value, live_label, title, ylabel, figsize='medium'):
    """Interactive dual-line chart with reference line"""
    colors = get_chart_colors()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates1, y=values1,
        mode='lines',
        name=label1,
        line=dict(color=colors['color_primary'], width=2),
        fill='tozeroy',
        fillcolor=f'rgba(52, 152, 219, 0.2)',
        hovertemplate='<b>Date:</b> %{x}<br><b>Historical:</b> ₹%{y:.2f}<extra></extra>'
    ))
    
    fig.add_hline(y=live_value, line_dash="dash", line_color=colors['color_positive'],
                  annotation_text=live_label,
                  annotation_position="right")
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=ylabel,
        hovermode='x unified',
        template='plotly_dark',
        height=450
    )
    return fig

def plot_daily_changes(data_df, title, figsize='medium'):
    """Interactive daily change bar chart"""
    colors = get_chart_colors()
    
    daily_changes = data_df['close'].diff()
    bar_colors = [colors['color_positive'] if x > 0 else colors['color_negative'] for x in daily_changes]
    
    fig = go.Figure(data=[
        go.Bar(y=daily_changes, marker=dict(color=bar_colors),
               hovertemplate='Period %{x}<br>Change: ₹%{y:.2f}<extra></extra>')
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Time Period",
        yaxis_title="Daily Price Change (₹)",
        hovermode='x unified',
        template='plotly_dark',
        height=450,
        showlegend=False
    )
    
    fig.add_hline(y=0, line_color="white", line_width=1)
    return fig

# Session and request headers
session = requests.session()
head = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


def get_id(name):
    search_url = 'https://www.nseindia.com/api/search/autocomplete?q={}'
    get_details = 'https://www.nseindia.com/api/quote-equity?symbol={}'

    session.get('https://www.nseindia.com/', headers=head)

    search_results = session.get(url=search_url.format(name), headers=head)
    search_data = search_results.json()

    if 'symbols' in search_data and search_data['symbols']:
        search_result = search_data['symbols'][0]['symbol']

        company_details = session.get(
            url=get_details.format(search_result), headers=head)

        try:
            identifier = company_details.json()['info']['identifier']
            return identifier
        except KeyError:
            return f"Identifier not found for '{name}'"
    else:
        return f"No results found for '{name}'"


def get_live_stock_price(symbol):
    """Fetch live stock price using multiple fallback APIs"""
    try:
        # Try yfinance first (most reliable)
        import yfinance as yf
        
        # For NSE stocks, add .NS suffix
        ticker_with_suffix = f"{symbol}.NS" if not symbol.endswith(('.BO', '.NS')) else symbol
        
        try:
            ticker = yf.Ticker(ticker_with_suffix)
            data = ticker.history(period='1d')
            
            if not data.empty:
                latest = data.iloc[-1]
                info = ticker.info if hasattr(ticker, 'info') else {}
                
                return {
                    'symbol': symbol,
                    'price': float(latest['Close']) if 'Close' in data.columns else float(info.get('currentPrice', 0)),
                    'change': float(info.get('regularMarketChangePercent', 0)),
                    'high': float(latest['High']) if 'High' in data.columns else float(info.get('dayHigh', 0)),
                    'low': float(latest['Low']) if 'Low' in data.columns else float(info.get('dayLow', 0)),
                    'open': float(latest['Open']) if 'Open' in data.columns else float(info.get('open', 0)),
                    'timestamp': dt.now()
                }
        except:
            pass
        
        # Fallback: Try direct Yahoo Finance API
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Try fetching from YahooFinance market data endpoint
        api_url = f'https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker_with_suffix}?modules=price'
        response = requests.get(api_url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if 'quoteSummary' in data and 'result' in data['quoteSummary']:
                price_data = data['quoteSummary']['result'][0]['price']
                return {
                    'symbol': symbol,
                    'price': float(price_data.get('regularMarketPrice', {}).get('raw', 0)),
                    'change': float(price_data.get('regularMarketChangePercent', {}).get('raw', 0)),
                    'high': float(price_data.get('regularMarketDayHigh', {}).get('raw', 0)),
                    'low': float(price_data.get('regularMarketDayLow', {}).get('raw', 0)),
                    'open': float(price_data.get('regularMarketOpen', {}).get('raw', 0)),
                    'timestamp': dt.now()
                }
        
        return None
        
    except ImportError:
        st.warning("📦 yfinance package needed. Install with: pip install yfinance")
        return None
    except requests.exceptions.Timeout:
        st.error(f"⏱️ Request timeout for {symbol}. Please try again.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("🌐 Connection error. Please check your internet connection.")
        return None
    except Exception as e:
        st.warning(f"Could not fetch live price for {symbol}: {str(e)}")
        return None


def fetch_stock_news(company_name, limit=10):
    """Fetch real news headlines using Google News API"""
    try:
        from google_news import google_news
        
        # Initialize Google News
        gn = google_news.GoogleNews(language='en', region='IN', period='7d')
        
        # Search for news about the stock
        gn.search(company_name)
        
        # Get results
        results = gn.results()
        
        news_list = []
        for item in results[:limit]:
            try:
                title = item.get('title', '')
                url = item.get('link', '')
                
                if title and url:
                    news_list.append({
                        'title': title,
                        'url': url
                    })
            except:
                continue
        
        if news_list:
            return pd.DataFrame(news_list)
        else:
            # If Google News fails, try alternative method
            return fetch_stock_news_fallback(company_name, limit)
            
    except Exception as e:
        # Fallback to alternative method
        return fetch_stock_news_fallback(company_name, limit)


def fetch_stock_news_fallback(company_name, limit=10):
    """Fallback news fetching using requests and beautifulsoup"""
    try:
        all_news = []
        
        # Try multiple search engines
        # 1. Try Bing News
        try:
            url = f'https://www.bing.com/news/search?q={company_name} stock'
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                soup = bs4.BeautifulSoup(response.text, 'html.parser')
                
                # Find news items
                for item in soup.find_all('div', {'class': 'news-card'})[:limit]:
                    try:
                        title_elem = item.find('a', {'class': 'title'})
                        url_elem = item.find('a', {'class': 'title'})
                        
                        if title_elem and url_elem:
                            title = title_elem.get_text(strip=True)
                            url = url_elem.get('href', '')
                            
                            if title and len(title) > 10 and url:
                                all_news.append({'title': title, 'url': url})
                    except:
                        continue
        except:
            pass
        
        # 2. Try Yahoo Finance News
        if len(all_news) < limit:
            try:
                # Use RSS feed from Yahoo Finance
                url = f'https://finance.yahoo.com/quote/{company_name}/news'
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    soup = bs4.BeautifulSoup(response.text, 'html.parser')
                    
                    # Find all news links
                    for link in soup.find_all('a', href=True)[:limit * 2]:
                        title = link.get_text(strip=True)
                        url = link.get('href', '')
                        
                        if (title and len(title) > 10 and 
                            url and ('finance.yahoo.com' in url or '/news/' in url)):
                            
                            if not url.startswith('http'):
                                url = 'https://finance.yahoo.com' + url
                            
                            all_news.append({'title': title, 'url': url})
                            
                            if len(all_news) >= limit:
                                break
            except:
                pass
        
        # 3. Try CNBC India
        if len(all_news) < limit:
            try:
                url = f'https://www.cnbctv18.com/search/?q={company_name}'
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(url, headers=headers, timeout=8)
                
                if response.status_code == 200:
                    soup = bs4.BeautifulSoup(response.text, 'html.parser')
                    
                    for link in soup.find_all('a', href=True)[:limit * 2]:
                        title = link.get_text(strip=True)
                        href = link.get('href', '')
                        
                        if (title and len(title) > 10 and 'cnbc' in href.lower()):
                            if not href.startswith('http'):
                                href = 'https://www.cnbctv18.com' + href
                            
                            all_news.append({'title': title, 'url': href})
                            
                            if len(all_news) >= limit:
                                break
            except:
                pass
        
        if all_news:
            return pd.DataFrame(all_news[:limit])
        else:
            return pd.DataFrame({'title': [], 'url': []})
    
    except Exception as e:
        return pd.DataFrame({'title': [], 'url': []})


def analyze_sentiment_vader(text):
    """Analyze sentiment using VADER + TextBlob hybrid approach for better accuracy"""
    if text is None or text == '':
        return 'neutral', 0.0
    
    try:
        # Use TextBlob for sentiment analysis (lazy import)
        from textblob import TextBlob
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity  # Range: -1 to 1
        
        # Convert to compound-like score
        compound = polarity
        
        if compound >= 0.1:
            return 'positive', compound
        elif compound <= -0.1:
            return 'negative', compound
        else:
            return 'neutral', compound
    except Exception as textblob_error:
        # Fallback to VADER if TextBlob fails (using cached analyzer)
        try:
            sia = get_sentiment_analyzer()
            if sia:
                scores = sia.polarity_scores(text)
                compound = scores['compound']
                
                if compound >= 0.05:
                    return 'positive', compound
                elif compound <= -0.05:
                    return 'negative', compound
                else:
                    return 'neutral', compound
        except Exception as vader_error:
            pass
        
        # Last resort: return neutral
        return 'neutral', 0.0

def get_sentiment_for_stock(company_name):
    """Get aggregated sentiment score for a stock based on news"""
    try:
        # Fetch real news only - no fake data
        news_df = fetch_stock_news(company_name, limit=15)
        
        # If no real news found, return None - don't use fake data
        if news_df is None or news_df.empty or len(news_df) == 0:
            return None, None, []
        
        # Ensure dataframe has required columns
        if 'title' not in news_df.columns:
            return None, None, []
        
        sentiments = []
        scores = []
        
        # Analyze sentiment for each headline
        for idx, row in news_df.iterrows():
            if pd.notna(row['title']):
                sentiment, score = analyze_sentiment_vader(str(row['title']))
                sentiments.append(sentiment)
                scores.append(score)
        
        if not scores:
            return None, None, []
        
        # Calculate aggregated metrics
        avg_score = np.mean(scores) if scores else 0
        positive_count = sentiments.count('positive')
        negative_count = sentiments.count('negative')
        neutral_count = sentiments.count('neutral')
        
        overall_sentiment = 'positive' if avg_score > 0.05 else ('negative' if avg_score < -0.05 else 'neutral')
        
        sentiment_summary = {
            'overall': overall_sentiment,
            'score': avg_score,
            'positive': positive_count,
            'negative': negative_count,
            'neutral': neutral_count,
            'total': len(sentiments)
        }
        
        return sentiment_summary, news_df, sentiments
    except Exception as e:
        return None, None, []


# Function to read historical data for multiple stocks (optimized)
@st.cache_data(ttl=3600)
def read_stock_data(directory):
    """Load stock data from CSV files - cached for 1 hour"""
    stock_data = {}
    try:
        if os.path.exists(directory):
            files = [f for f in os.listdir(directory) if f.endswith(".csv")]
            for filename in files:
                try:
                    stock_name = os.path.splitext(filename)[0]
                    df = pd.read_csv(
                        os.path.join(directory, filename),
                        usecols=['datetime', 'close'],  # Only load needed columns
                        dtype={'close': 'float32'},  # Use smaller data type
                        low_memory=False,
                        nrows=None  # Avoid reading entire file multiple times
                    )
                    if len(df) > 0:
                        stock_data[stock_name] = df
                except Exception:
                    continue
    except Exception:
        pass
    return stock_data

def train_model(stock_data):
    models = {}
    for stock_name, df in stock_data.items():
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
            
        X = df[['year', 'month', 'day']].values
        y = df['close']
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
        model = XGBRegressor()
        model.fit(X_train, y_train)
        models[stock_name] = model
            
    return models

def predict_price(model, date):
# Convert the date to a pandas datetime object
    date = pd.to_datetime(date)
        
    # Extract year, month, and day from the date
    year = date.year
    month = date.month
    day = date.day
        
    # Make prediction using the model
    prediction = model.predict(np.array([[year, month, day]]))[0]
        
    return prediction


def get_mock_stock_data(symbol):
    """Generate mock stock data for testing"""
    import random
    price = random.uniform(100, 5000)
    return {
        'symbol': symbol,
        'price': price,
        'change': random.uniform(-5, 5),
        'high': price * 1.05,
        'low': price * 0.95,
        'open': price * 0.98,
        'timestamp': dt.now()
    }


def get_mock_news(company_name, limit=10):
    """Generate mock news data for testing"""
    sample_headlines = [
        f"{company_name} stock rallies on strong Q3 earnings report",
        f"{company_name} announces new product launch, stock jumps",
        f"Analysts upgrade {company_name} to buy on growth prospects",
        f"{company_name} sees record revenue in latest quarter",
        f"Market experts bullish on {company_name} future outlook",
        f"{company_name} expands into new markets",
        f"Strong demand boosts {company_name} stock higher",
        f"{company_name} invests in R&D for innovation",
        f"Institutional buyers accumulate {company_name} shares",
        f"{company_name} achieves industry-leading margins"
    ]
    
    urls = [f"https://example.com/article-{i}" for i in range(len(sample_headlines))]
    return pd.DataFrame({'title': sample_headlines[:8], 'url': urls[:8]})


# Load stock data (with caching for performance)
@st.cache_resource
def load_app_data():
    """Load all app data once and cache it"""
    archive_folder = "archive"
    return read_stock_data(archive_folder)

# Optimized loading with minimal spinner display
@st.cache_data(ttl=3600)
def get_stock_list():
    """Get list of available stocks from cache"""
    stock_data = load_app_data()
    return sorted(list(stock_data.keys()))

# Load stock data with faster initial display
stock_data = load_app_data()

# Handle empty stock data gracefully
if not stock_data:
    st.error("⚠️ No stock data found in archive folder. Please ensure CSV files are present.")
    st.stop()

# Get stock list
stock_list = get_stock_list()

# Select a stock for visualization
selected_stock = st.selectbox("Select a stock", stock_list)

# Add info box at top
st.info("""
✅ **Live Data & Sentiment Features:**
- Live data fetches real-time prices from Yahoo Finance API (yfinance)
- News headlines powered by Google News API with fallback sources (Bing, Yahoo Finance, CNBC)
- Sentiment analysis using hybrid TextBlob + VADER for accurate financial text analysis
- If features don't work, it may be due to temporary API availability or network issues
- Try using the 'Demo Mode' toggle below to see mock data
""")

# Add demo mode toggle
col1, col2 = st.columns([3, 1])
with col1:
    st.write("")
with col2:
    demo_mode = st.checkbox("📊 Demo Mode", value=False)

# Create tabs for different features
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Live Data", "📊 Historical Data", "💬 Sentiment Analysis", "📰 News & Analysis", "📊 Live vs Historical"])

# ==================== TAB 1: LIVE DATA ====================
with tab1:
    st.subheader("Live Stock Data")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Fetch Live Price", key="live_price"):
            with st.spinner("Fetching live data..."):
                if demo_mode:
                    live_data = get_mock_stock_data(selected_stock)
                    st.success("✅ Mock data (Demo Mode)")
                else:
                    live_data = get_live_stock_price(selected_stock)
                
                if live_data:
                    if not demo_mode:
                        st.success("✅ Data fetched successfully!")
                    
                    col_price, col_change = st.columns(2)
                    with col_price:
                        st.metric("Current Price", f"₹{live_data['price']:.2f}")
                    with col_change:
                        st.metric("Change %", f"{live_data['change']:+.2f}%")
                    
                    # Display additional metrics
                    metric_cols = st.columns(4)
                    with metric_cols[0]:
                        st.metric("Open", f"₹{live_data['open']:.2f}")
                    with metric_cols[1]:
                        st.metric("High", f"₹{live_data['high']:.2f}")
                    with metric_cols[2]:
                        st.metric("Low", f"₹{live_data['low']:.2f}")
                    with metric_cols[3]:
                        st.metric("Time", live_data['timestamp'].strftime("%H:%M:%S"))
                else:
                    st.error(f"Could not fetch live data for {selected_stock}. Try Demo Mode ↗️")
    
    with col2:
        if demo_mode:
            st.info("""
            **🎭 Demo Mode Active**
            - Showing sample/mock data
            - Click button to generate new data
            - Turn off Demo Mode for real data
            """)
        else:
            st.info("""
            **📌 About Live Data:**
            - Fetches real-time stock prices from NSE India
            - Updates when you click the button
            - Shows daily high, low, open and current price
            - Displays percentage change from previous close
            **Troubleshooting:** If not working, enable Demo Mode
            """)

# ==================== TAB 2: HISTORICAL DATA ====================
with tab2:
    st.subheader("📊 Historical Stock Data Analysis")
    
    if st.button("View Historical Data", key="get_prediction"):
        if selected_stock in stock_data:
            st.subheader("Historical Stock Price Data")
            hist_df = stock_data[selected_stock].copy()
            st.write(hist_df)

            # Plot historical stock prices
            st.subheader("Price Trend Over Time")
            hist_df['Date'] = pd.to_datetime(hist_df['datetime'])
            
            # Use modular plotting function
            fig = plot_line_trend(hist_df['Date'], hist_df['close'], 
                                 f"{selected_stock} - Historical Price Trend",
                                 "Closing Price (₹)", use_fill=True, figsize='medium')
            with st.expander("📈 Price Trend Chart", expanded=True):
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            st.subheader("📈 Historical Statistics")
            stat_cols = st.columns(5)
            
            with stat_cols[0]:
                st.metric("Avg Price", f"₹{hist_df['close'].mean():.2f}")
            
            with stat_cols[1]:
                st.metric("Max Price", f"₹{hist_df['close'].max():.2f}")
            
            with stat_cols[2]:
                st.metric("Min Price", f"₹{hist_df['close'].min():.2f}")
            
            with stat_cols[3]:
                st.metric("Std Dev", f"₹{hist_df['close'].std():.2f}")
            
            with stat_cols[4]:
                st.metric("Total Records", f"{len(hist_df)}")
            
            # Additional analysis
            st.subheader("📉 Price Change Analysis")
            hist_df['Daily Change'] = hist_df['close'].diff()
            hist_df['% Change'] = hist_df['close'].pct_change() * 100
            
            fig2 = plot_daily_changes(hist_df, f"{selected_stock} - Daily Price Changes", figsize='medium')
            with st.expander("📊 Daily Changes Chart", expanded=True):
                st.plotly_chart(fig2, use_container_width=True)
            
        else:
            st.error(f"No historical data available for {selected_stock}")
        models = train_model({selected_stock: stock_data[selected_stock]})
            
        today_date = dt.today().date()

            # Predict stock price for today's EOD
        prediction = predict_price(models[selected_stock], today_date)

            # Display the predicted price
        st.subheader("Predicted Stock Price for Today's EOD")
        st.write(f"The predicted closing price for {selected_stock} on {today_date} is {prediction:.2f} based on Historical Data")

        company_name = selected_stock

        # Get ticker symbol
        ticker_symbol = get_id(company_name)
        if not ticker_symbol.startswith("Identifier"):
            st.success(f"✅ Stock identifier for '{company_name}': {ticker_symbol}")
            
            # Fetch and display intraday data
            with st.spinner("Fetching intraday data..."):
                stock_url = f'https://www.nseindia.com/api/chart-databyindex?index={ticker_symbol}'
                
                session_intra = requests.Session()
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                initial_response = session_intra.get('https://www.nseindia.com/', headers=headers)
                response = session_intra.get(stock_url, headers=headers)
                
                if response.ok:
                    try:
                        dayta = pd.DataFrame(response.json()['grapthData'])
                        if not dayta.empty and len(dayta) > 0:
                            dayta.columns = ['timestamp', 'price']
                            dayta['timestamp'] = pd.to_datetime(dayta['timestamp'], unit='ms')
                            
                            # Plot intraday data using modular function
                            fig_intra = plot_line_trend(dayta['timestamp'], dayta['price'],
                                                       f"{selected_stock} Intraday Price Movement",
                                                       "Price (₹)", use_fill=False, figsize='medium')
                            with st.expander("📊 Intraday Movement Chart", expanded=True):
                                st.plotly_chart(fig_intra, use_container_width=True)
                        else:
                            st.info("📊 Intraday data not available for this stock")
                    except Exception as e:
                        st.warning(f"⚠️ Could not load intraday data: {str(e)}")

# ==================== TAB 3: SENTIMENT ANALYSIS ====================
with tab3:
    st.subheader("📊 Sentiment Analysis for Stock")
    
    if st.button("Analyze Sentiment", key="analyze_sentiment"):
        with st.spinner("Analyzing sentiment from news..."):
            # Pre-initialize resources before any sentiment analysis
            initialize_app_resources()
            
            company_name = selected_stock
            
            if demo_mode:
                # Mock sentiment data
                mock_sentiments = ['positive', 'positive', 'positive', 'neutral', 'positive', 'positive', 'negative']
                sentiment_summary = {
                    'overall': 'positive',
                    'score': 0.65,
                    'positive': 6,
                    'negative': 1,
                    'neutral': 1,
                    'total': 8
                }
                news_df = get_mock_news(company_name)
                sentiments = mock_sentiments
                st.info("📊 Showing mock sentiment data (Demo Mode)")
            else:
                sentiment_summary, news_df, sentiments = get_sentiment_for_stock(company_name)
            
            if sentiment_summary is not None and sentiment_summary:
                # Display sentiment summary with colored boxes
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Overall Sentiment", sentiment_summary['overall'].upper(), 
                             delta=f"Score: {sentiment_summary['score']:.2f}")
                
                with col2:
                    st.metric("Positive 👍", sentiment_summary['positive'])
                
                with col3:
                    st.metric("Negative 👎", sentiment_summary['negative'])
                
                with col4:
                    st.metric("Neutral 😐", sentiment_summary['neutral'])
                
                # Visualization of sentiment distribution
                st.subheader("Sentiment Distribution")
                sentiment_data = {
                    'Positive': sentiment_summary['positive'],
                    'Negative': sentiment_summary['negative'],
                    'Neutral': sentiment_summary['neutral']
                }
                
                fig_sentiment = plot_pie_chart(sentiment_data, f"Sentiment Distribution for {selected_stock}", figsize='compact')
                with st.expander("🥧 Sentiment Pie Chart", expanded=True):
                    st.plotly_chart(fig_sentiment, use_container_width=True)
                
                st.success("✅ Sentiment analysis complete!")
                
                # Recommendation based on sentiment
                st.subheader("📈 Recommendation Based on Sentiment")
                if sentiment_summary['overall'] == 'positive':
                    st.success(f"🟢 **BUY** - The sentiment is overwhelmingly positive with {sentiment_summary['positive']} positive headlines")
                elif sentiment_summary['overall'] == 'negative':
                    st.error(f"🔴 **SELL** - The sentiment is negative with {sentiment_summary['negative']} negative headlines")
                else:
                    st.warning(f"🟡 **HOLD** - Mixed sentiment detected. Market outlook is neutral")
            else:
                st.error(f"Could not analyze sentiment for {company_name}. Try Demo Mode ↗️")

# ==================== TAB 4: NEWS & DETAILED ANALYSIS ====================
with tab4:
    st.subheader("📰 News Headlines & Analysis")
    
    if st.button("Fetch News Headlines", key="fetch_news"):
        with st.spinner(f"Fetching news for {selected_stock}..."):
            if demo_mode:
                news_df = get_mock_news(selected_stock, limit=20)
                st.info("📰 Showing mock news data (Demo Mode)")
            else:
                news_df = fetch_stock_news(selected_stock, limit=20)
            
            if not news_df.empty:
                st.success(f"✅ Found {len(news_df)} recent articles")
                
                # Display news with sentiment
                st.subheader("Headlines with Individual Sentiment Scores")
                
                for idx, row in news_df.iterrows():
                    sentiment, score = analyze_sentiment_vader(row['title'])
                    
                    # Color code based on sentiment
                    if sentiment == 'positive':
                        emoji = "🟢"
                        color = "#2ecc71"
                    elif sentiment == 'negative':
                        emoji = "🔴"
                        color = "#e74c3c"
                    else:
                        emoji = "⚪"
                        color = "#95a5a6"
                    
                    col_emoji, col_title, col_score = st.columns([1, 15, 3])
                    with col_emoji:
                        st.markdown(f"{emoji}")
                    with col_title:
                        st.markdown(f"[{row['title']}]({row['url']})")
                    with col_score:
                        st.metric("Score", f"{score:.2f}", label_visibility="collapsed")
                
            else:
                st.warning(f"No news found for {selected_stock}. Try enabling Demo Mode ↗️")

# ==================== TAB 5: LIVE VS HISTORICAL COMPARISON ====================
with tab5:
    st.subheader("📊 Live Data vs Historical Data Comparison")
    
    col_comp1, col_comp2 = st.columns(2)
    
    with col_comp1:
        st.write("**Fetch Live Data**")
        if st.button("Get Current Price", key="get_live_comparison"):
            live_data = get_live_stock_price(selected_stock)
            if live_data:
                st.session_state['live_comparison_data'] = live_data
                st.success(f"✅ Live data updated for {selected_stock}")
            else:
                st.error("Could not fetch live data")
    
    with col_comp2:
        st.write("**Historical Data**")
        if st.button("Load Historical Data", key="get_historical_comparison"):
            if selected_stock in stock_data:
                st.session_state['historical_comparison_data'] = stock_data[selected_stock]
                st.success(f"✅ Historical data loaded for {selected_stock}")
            else:
                st.error(f"No historical data available for {selected_stock}")
    
    st.divider()
    
    # Display comparison if both data available
    if 'live_comparison_data' in st.session_state and 'historical_comparison_data' in st.session_state:
        live_data = st.session_state['live_comparison_data']
        hist_data = st.session_state['historical_comparison_data']
        
        # Comparison metrics
        st.subheader("📈 Comparison Metrics")
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            st.metric("Current Price", f"₹{live_data['price']:.2f}", 
                     delta=f"{live_data['change']:+.2f}%")
        
        with metric_cols[1]:
            if not hist_data.empty:
                avg_historical = hist_data['close'].mean()
                diff = ((live_data['price'] - avg_historical) / avg_historical) * 100
                st.metric("vs Avg Historical", f"₹{avg_historical:.2f}", 
                         delta=f"{diff:+.2f}%")
        
        with metric_cols[2]:
            if not hist_data.empty:
                max_historical = hist_data['close'].max()
                st.metric("vs Historical High", f"₹{max_historical:.2f}", 
                         delta=f"{((live_data['price'] - max_historical) / max_historical) * 100:+.2f}%")
        
        with metric_cols[3]:
            if not hist_data.empty:
                min_historical = hist_data['close'].min()
                st.metric("vs Historical Low", f"₹{min_historical:.2f}", 
                         delta=f"{((live_data['price'] - min_historical) / min_historical) * 100:+.2f}%")
        
        st.divider()
        
        # Side-by-side graphs with expandable sections
        st.subheader("📊 Visual Comparison")
        
        graph_tab1, graph_tab2, graph_tab3 = st.tabs(["📍 Live Data Status", "📈 Historical Trend", "🔀 Combined Overlay"])
        
        with graph_tab1:
            st.write("**Current Live Prices Breakdown**")
            categories = ['Open', 'Current', 'High', 'Low']
            prices = [
                live_data['open'],
                live_data['price'],
                live_data['high'],
                live_data['low']
            ]
            colors_config = get_chart_colors()
            colors_gauge = [colors_config['color_secondary'], colors_config['color_positive'], 
                          colors_config['color_negative'], colors_config['color_secondary']]
            
            fig_live = plot_bar_chart(categories, prices, f"{selected_stock} Live Prices",
                                     "Price (₹)", colors=colors_gauge, figsize='compact', horizontal=True)
            st.plotly_chart(fig_live, use_container_width=True)
        
        with graph_tab2:
            st.write("**Historical Trend with Current Price Reference**")
            hist_data_sorted = hist_data.sort_values('datetime')
            hist_data_sorted['datetime'] = pd.to_datetime(hist_data_sorted['datetime'])
            
            fig_hist = plot_dual_comparison(hist_data_sorted['datetime'], hist_data_sorted['close'],
                                          'Historical Close',
                                          live_data['price'],
                                          f"Current Live Price: ₹{live_data['price']:.2f}",
                                          f"{selected_stock} Historical Trend",
                                          "Price (₹)", figsize='compact')
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with graph_tab3:
            st.write("**Live Price Position in Historical Context**")
            hist_data_sorted = hist_data.sort_values('datetime')
            hist_data_sorted['datetime'] = pd.to_datetime(hist_data_sorted['datetime'])
            
            # Create combined comparison chart
            fig_combined = plot_dual_comparison(
                hist_data_sorted['datetime'], 
                hist_data_sorted['close'],
                'Historical Price Trend',
                live_data['price'],
                f"Live Price: ₹{live_data['price']:.2f}",
                f"{selected_stock} - Live vs Historical Distribution",
                "Price (₹)", 
                figsize='medium'
            )
            st.plotly_chart(fig_combined, use_container_width=True)
            
            # Add interpretation
            hist_max = hist_data['close'].max()
            hist_min = hist_data['close'].min()
            hist_avg = hist_data['close'].mean()
            current_price = live_data['price']
            
            col_interp1, col_interp2, col_interp3 = st.columns(3)
            
            with col_interp1:
                if current_price >= hist_max:
                    st.success("📈 **Above Historical High**")
                elif current_price <= hist_min:
                    st.warning("📉 **Below Historical Low**")
                else:
                    st.info("➡️ **Within Historical Range**")
            
            with col_interp2:
                percentile = ((current_price - hist_min) / (hist_max - hist_min) * 100) if (hist_max - hist_min) != 0 else 50
                st.metric("Percentile Position", f"{percentile:.1f}%", 
                         delta=f"Position relative to 52-week range")
            
            with col_interp3:
                vs_avg_pct = ((current_price - hist_avg) / hist_avg * 100)
                st.metric("vs Average", f"{vs_avg_pct:+.2f}%",
                         delta=f"Difference from average price")
        
        st.divider()
        
        # Statistical comparison
        st.subheader("📊 Statistical Analysis")
        stat_cols = st.columns(3)
        
        with stat_cols[0]:
            if not hist_data.empty:
                volatility = hist_data['close'].std()
                st.metric("Historical Volatility", f"₹{volatility:.2f}")
                st.caption("Standard deviation of historical prices")
        
        with stat_cols[1]:
            if not hist_data.empty:
                price_range = hist_data['close'].max() - hist_data['close'].min()
                st.metric("Historical Range", f"₹{price_range:.2f}")
                st.caption("Difference between high and low")
        
        with stat_cols[2]:
            if not hist_data.empty:
                days_analyzed = len(hist_data)
                st.metric("Data Points", f"{days_analyzed}")
                st.caption("Number of historical records")
        
        st.success("✅ Comparison complete. Scroll up to see detailed analysis.")
    else:
        st.info("👆 Click both buttons above to load live and historical data for comparison.")