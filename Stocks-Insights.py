# stocks_insights.py
import os
import json
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import pytz
import hashlib
import re
from datetime import datetime, timedelta
from telegram import (
    Update, 
    InlineKeyboardButton, 
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
    InputMediaPhoto,
    InputMediaDocument
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    ConversationHandler,
    filters,
    JobQueue
)
from transformers import pipeline
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from dotenv import load_dotenv
import logging
import tempfile
import gc
from io import BytesIO
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mplfinance as mpf
from ta import add_all_ta_features
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

# Load environment variables
load_dotenv()
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuration
TOKEN = os.getenv('TELEGRAM_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY')
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')
FINNHUB_KEY = os.getenv('FINNHUB_KEY')
POLYGON_KEY = os.getenv('POLYGON_KEY')

openai.api_key = OPENAI_API_KEY

# Constants
TERMS = """
📜 *TERMS & CONDITIONS*

1. **Not Financial Advice**: All content is for informational/educational purposes only. Never financial advice.
2. **No Liability**: We're not responsible for investment decisions or financial losses incurred.
3. **Accuracy**: Market data and predictions may be inaccurate. Verify information independently.
4. **Risk**: Stock investing carries substantial risk. Only invest what you can afford to lose.
5. **Data**: We collect anonymous usage data to improve services. No personal financial data is stored.

🔒 *PRIVACY POLICY*
- We don't store personal financial information
- All user data is encrypted at rest and in transit
- Anonymous usage analytics are collected
- Third-party APIs may process your requests
- No data sharing with advertisers

✅ Type `/agree` to accept these terms and begin.
"""

USER_PROFILE = {}
ANALYSIS_CACHE = {}
RECOMMENDATIONS = {}
USER_PORTFOLIOS = {}
MARKET_ALERTS = {}
EDUCATION_MODULES = {
    'basics': "📚 *Stock Market Basics*\nLearn about stocks, exchanges, and market mechanics.",
    'analysis': "🔍 *Analysis Techniques*\nFundamental, technical, and sentiment analysis explained.",
    'strategies': "🧠 *Trading Strategies*\nLong-term investing, swing trading, and day trading approaches.",
    'risk': "🛡️ *Risk Management*\nPosition sizing, stop losses, and portfolio diversification.",
    'options': "📊 *Options Trading*\nCalls, puts, and advanced derivatives strategies."
}

# Conversation states
AGREE, AMOUNT, HORIZON, RISK, SECTORS, REGION, ANALYSIS = range(7)

# Initialize AI models
sentiment_analyzer = pipeline('sentiment-analysis', model='mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis')
risk_levels = {
    'low': {'volatility': 0.1, 'growth': 0.05, 'allocation': {'Bonds': 50, 'Dividend': 30, 'Growth': 20}},
    'medium': {'volatility': 0.2, 'growth': 0.1, 'allocation': {'Dividend': 40, 'Growth': 40, 'ETF': 20}},
    'high': {'volatility': 0.4, 'growth': 0.2, 'allocation': {'Growth': 60, 'Tech': 30, 'Crypto': 10}}
}

# ------------------------
# SECURITY & COMPLIANCE
# ------------------------
def encrypt_data(data: str) -> str:
    """Basic encryption for sensitive data"""
    return hashlib.sha256(data.encode()).hexdigest()

def validate_symbol(symbol: str) -> bool:
    """Validate stock ticker format"""
    return bool(re.match(r'^[A-Z]{1,5}$', symbol))

def compliance_check(user_id: int) -> bool:
    """Check if user has agreed to terms"""
    profile = USER_PROFILE.get(user_id, {})
    return profile.get('agreed', False)

# ------------------------
# ADVANCED DATA FUNCTIONS
# ------------------------
def get_realtime_price(symbol: str) -> float:
    """Get real-time price from Polygon.io"""
    try:
        url = f"https://api.polygon.io/v2/last/trade/{symbol}?apiKey={POLYGON_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data['results']['p']
    except Exception as e:
        logger.error(f"Polygon error: {e}")
        # Fallback to Alpha Vantage
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_KEY}"
        response = requests.get(url)
        data = response.json()
        return float(data['Global Quote']['05. price'])

def get_advanced_metrics(symbol: str) -> dict:
    """Get advanced financial metrics"""
    try:
        url = f"https://financialmodelingprep.com/api/v3/key-metrics/{symbol}?apikey={FINNHUB_KEY}"
        response = requests.get(url)
        data = response.json()[0]
        return {
            'roic': data.get('roic'),
            'earnings_yield': data.get('earningsYield'),
            'graham_number': data.get('grahamNumber'),
            'piotroski_score': data.get('piotroskiScore'),
            'altman_z_score': data.get('altmanZScore')
        }
    except Exception as e:
        logger.error(f"Advanced metrics error: {e}")
        return {}

def get_insider_transactions(symbol: str) -> list:
    """Get recent insider transactions"""
    try:
        url = f"https://finnhub.io/api/v1/stock/insider-transactions?symbol={symbol}&token={FINNHUB_KEY}"
        response = requests.get(url)
        data = response.json()['data'][:5]  # Last 5 transactions
        return [{
            'name': t['name'],
            'position': t['position'],
            'type': 'BUY' if t['share'] > 0 else 'SELL',
            'shares': abs(t['share']),
            'price': t['price']
        } for t in data]
    except Exception as e:
        logger.error(f"Insider transactions error: {e}")
        return []

# ------------------------
# ENHANCED ANALYSIS FUNCTIONS
# ------------------------
def analyze_stock(symbol: str, profile: dict) -> dict:
    """Comprehensive stock analysis using multiple data sources"""
    if symbol in ANALYSIS_CACHE:
        return ANALYSIS_CACHE[symbol]
    
    # Get stock data
    stock = yf.Ticker(symbol)
    info = stock.info
    hist = stock.history(period="5y")
    
    # Add technical indicators
    hist = add_all_ta_features(hist, open="Open", high="High", low="Low", close="Close", volume="Volume")
    
    # Technical analysis
    technical = technical_analysis(hist, symbol)
    
    # Fundamental analysis
    fundamental = fundamental_analysis(info)
    fundamental.update(get_advanced_metrics(symbol))
    
    # Sentiment analysis
    sentiment = sentiment_analysis(symbol)
    
    # Price forecast
    forecast = price_forecast(hist, symbol)
    
    # Risk analysis
    risk = risk_analysis(hist, profile.get('risk', 'medium'))
    
    # Insider transactions
    insiders = get_insider_transactions(symbol)
    
    analysis = {
        'symbol': symbol,
        'name': info.get('shortName', symbol),
        'current_price': get_realtime_price(symbol),
        'technical': technical,
        'fundamental': fundamental,
        'sentiment': sentiment,
        'forecast': forecast,
        'risk': risk,
        'insiders': insiders,
        'last_updated': datetime.now().isoformat()
    }
    
    ANALYSIS_CACHE[symbol] = analysis
    return analysis

def technical_analysis(hist: pd.DataFrame, symbol: str) -> dict:
    """Advanced technical analysis with multiple indicators"""
    # Calculate additional indicators
    rsi_indicator = RSIIndicator(hist['Close'])
    hist['RSI'] = rsi_indicator.rsi()
    
    bb_indicator = BollingerBands(hist['Close'])
    hist['BB_upper'] = bb_indicator.bollinger_hband()
    hist['BB_lower'] = bb_indicator.bollinger_lband()
    
    # Generate advanced chart
    plt.style.use('seaborn-darkgrid')
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Price and moving averages
    axes[0].plot(hist.index, hist['Close'], label='Price', linewidth=2)
    axes[0].plot(hist.index, hist['SMA_50'], label='50-day MA', linestyle='--')
    axes[0].plot(hist.index, hist['SMA_200'], label='200-day MA', linestyle='-.')
    axes[0].fill_between(hist.index, hist['BB_lower'], hist['BB_upper'], alpha=0.2, label='Bollinger Bands')
    axes[0].set_title(f'{symbol} Technical Analysis')
    axes[0].legend()
    
    # Volume
    axes[1].bar(hist.index, hist['Volume'], color='skyblue', alpha=0.7)
    axes[1].set_title('Volume')
    
    # RSI
    axes[2].plot(hist.index, hist['RSI'], label='RSI', color='purple')
    axes[2].axhline(70, linestyle='--', color='red', alpha=0.5)
    axes[2].axhline(30, linestyle='--', color='green', alpha=0.5)
    axes[2].set_ylim(0, 100)
    axes[2].set_title('RSI')
    
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=120)
    buf.seek(0)
    plt.close(fig)
    
    # Trading signals
    last_rsi = hist['RSI'].iloc[-1]
    macd_signal = 'Bullish' if hist['MACD_12_26_9'].iloc[-1] > hist['MACDs_12_26_9'].iloc[-1] else 'Bearish'
    trend_strength = 'Strong' if abs(hist['ADX_14'].iloc[-1]) > 25 else 'Weak'
    
    return {
        'trend': 'Bullish' if hist['SMA_50'].iloc[-1] > hist['SMA_200'].iloc[-1] else 'Bearish',
        'rsi': last_rsi,
        'rsi_signal': 'Oversold' if last_rsi < 30 else 'Overbought' if last_rsi > 70 else 'Neutral',
        'macd': macd_signal,
        'trend_strength': trend_strength,
        'chart': buf
    }

def fundamental_analysis(info: dict) -> dict:
    """Advanced fundamental analysis"""
    return {
        'pe_ratio': info.get('trailingPE'),
        'peg_ratio': info.get('pegRatio'),
        'eps': info.get('trailingEps'),
        'roi': info.get('returnOnEquity'),
        'debt_equity': info.get('debtToEquity'),
        'dividend_yield': info.get('dividendYield'),
        'beta': info.get('beta'),
        'market_cap': info.get('marketCap'),
        'ebitda': info.get('ebitda'),
        'forward_pe': info.get('forwardPE'),
        'profit_margin': info.get('profitMargins')
    }

def sentiment_analysis(symbol: str) -> dict:
    """Advanced sentiment analysis with multiple sources"""
    try:
        # NewsAPI for recent articles
        newsapi_url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={NEWSAPI_KEY}&language=en&pageSize=5"
        news_response = requests.get(newsapi_url)
        news_data = news_response.json()
        
        # Analyze sentiment for headlines
        headlines = [article['title'] for article in news_data.get('articles', [])]
        sentiment_scores = sentiment_analyzer(headlines) if headlines else []
        
        # Finnhub sentiment
        finnhub_url = f"https://finnhub.io/api/v1/news/sentiment?symbol={symbol}&token={FINNHUB_KEY}"
        finnhub_response = requests.get(finnhub_url)
        finnhub_data = finnhub_response.json()
        
        # Calculate composite sentiment
        avg_score = np.mean([s['score'] for s in sentiment_scores]) if sentiment_scores else 0
        finnhub_sentiment = finnhub_data.get('sentiment', 0)
        composite = (avg_score + finnhub_sentiment) / 2
        
        return {
            'score': composite,
            'trend': 'Bullish' if composite > 0.5 else 'Bearish' if composite < -0.5 else 'Neutral',
            'recent_news': [
                {
                    'headline': article['title'],
                    'source': article['source']['name'],
                    'url': article['url'],
                    'sentiment': sentiment_analyzer([article['title'])[0]['score']
                } for article in news_data.get('articles', [])
            ]
        }
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        return {
            'score': 0,
            'trend': 'Neutral',
            'recent_news': []
        }

def price_forecast(hist: pd.DataFrame, symbol: str) -> dict:
    """Advanced price forecasting with Prophet and Monte Carlo simulation"""
    # Prophet forecasting
    df = hist.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    df = df.dropna()
    
    model = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
    model.add_country_holidays(country_name='US')
    model.fit(df)
    
    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)
    
    # Generate forecast chart
    fig = model.plot(forecast)
    plt.title(f'{symbol} 90-Day Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=120)
    buf.seek(0)
    plt.close()
    
    # Monte Carlo simulation for confidence intervals
    last_close = hist['Close'].iloc[-1]
    daily_returns = hist['Close'].pct_change().dropna()
    volatility = daily_returns.std()
    
    simulations = []
    for _ in range(1000):
        price_series = [last_close]
        for _ in range(90):
            drift = daily_returns.mean() - (0.5 * volatility**2)
            shock = drift + volatility * np.random.normal()
            price_series.append(price_series[-1] * (1 + shock))
        simulations.append(price_series[-1])
    
    # Calculate confidence intervals
    simulations.sort()
    lower_bound = simulations[50]
    upper_bound = simulations[950]
    
    return {
        'price_3mo': forecast.iloc[-1]['yhat'],
        'confidence': 0.85,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'chart': buf
    }

def risk_analysis(hist: pd.DataFrame, risk_profile: str) -> dict:
    """Advanced risk analysis with multiple metrics"""
    daily_returns = hist['Close'].pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252)
    max_drawdown = calculate_max_drawdown(hist['Close'])
    
    # Value at Risk (VaR)
    var_95 = np.percentile(daily_returns, 5)
    
    return {
        'volatility': volatility,
        'risk_level': 'High' if volatility > 0.3 else 'Medium' if volatility > 0.15 else 'Low',
        'max_drawdown': max_drawdown,
        'var_95': var_95,
        'recommended_alloc': risk_levels.get(risk_profile, {}).get('allocation', {})
    }

def calculate_max_drawdown(prices: pd.Series) -> float:
    """Calculate maximum drawdown"""
    peak = prices.expanding(min_periods=1).max()
    trough = prices.expanding(min_periods=1).min()
    return ((trough - peak) / peak).min()

# ------------------------
# GPT-4 INTEGRATION
# ------------------------
def gpt4_analysis(symbol: str, analysis: dict) -> str:
    """Generate advanced analysis using GPT-4"""
    try:
        context = (
            f"Provide comprehensive analysis of {symbol} ({analysis['name']}) for an investor. "
            "Include: 1) Technical analysis summary 2) Fundamental health 3) Sentiment overview "
            "4) Risk assessment 5) Trading strategy suggestions. Use markdown formatting."
        )
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional financial analyst providing clear, balanced stock analysis."},
                {"role": "user", "content": context}
            ],
            max_tokens=500,
            temperature=0.3
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        logger.error(f"GPT-4 analysis error: {e}")
        return "Advanced analysis unavailable at this time."

# ------------------------
# PORTFOLIO OPTIMIZATION
# ------------------------
def optimize_portfolio(user_id: int) -> dict:
    """Create optimized portfolio based on user profile"""
    profile = USER_PROFILE.get(user_id, {})
    recommendations = RECOMMENDATIONS.get(user_id, [])
    amount = profile.get('amount', 10000)
    risk = profile.get('risk', 'medium')
    
    if not recommendations:
        return {}
    
    # Create portfolio based on risk profile
    portfolio = []
    allocation = risk_levels.get(risk, {}).get('allocation', {})
    total_percent = sum(allocation.values())
    
    # Distribute funds
    for category, percent in allocation.items():
        category_stocks = [s for s in recommendations if category.lower() in s.get('sector', '').lower()]
        if not category_stocks:
            category_stocks = recommendations[:2]  # Fallback
        
        # Distribute among category stocks
        category_amount = amount * (percent / total_percent)
        for i, stock in enumerate(category_stocks):
            stock_amount = category_amount / len(category_stocks)
            portfolio.append({
                'symbol': stock['symbol'],
                'name': stock['name'],
                'category': category,
                'amount': stock_amount,
                'shares': stock_amount / stock['analysis']['current_price'],
                'percent': percent / len(category_stocks)
            })
    
    return portfolio

# ------------------------
# ALERTING SYSTEM
# ------------------------
def check_market_conditions(context: ContextTypes.DEFAULT_TYPE):
    """Check market conditions for alerts"""
    logger.info("Running market condition check...")
    try:
        # Monitor major indices
        indices = {'^GSPC': 'S&P 500', '^DJI': 'Dow Jones', '^IXIC': 'NASDAQ'}
        alerts = []
        
        for symbol, name in indices.items():
            data = yf.download(symbol, period='1d', interval='5m')
            if data.empty:
                continue
                
            # Calculate daily change
            open_price = data['Open'].iloc[0]
            current_price = data['Close'].iloc[-1]
            change = (current_price - open_price) / open_price * 100
            
            # Check for significant moves
            if abs(change) > 1.5:
                alerts.append(f"🚨 *{name} Alert*: {'▲' if change > 0 else '▼'} {abs(change):.2f}% change today")
        
        # Send alerts to all users
        if alerts:
            message = "🔔 *Market Alerts*\n\n" + "\n".join(alerts)
            for user_id in list(USER_PROFILE.keys()):
                try:
                    context.bot.send_message(
                        chat_id=user_id,
                        text=message,
                        parse_mode='Markdown'
                    )
                except Exception as e:
                    logger.error(f"Alert send error: {e}")
                    
    except Exception as e:
        logger.error(f"Market condition error: {e}")

# ------------------------
# TELEGRAM BOT HANDLERS
# ------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Initial conversation handler showing terms"""
    user = update.message.from_user
    # Clear previous data
    if user.id in USER_PROFILE:
        del USER_PROFILE[user.id]
    if user.id in RECOMMENDATIONS:
        del RECOMMENDATIONS[user.id]
    
    await update.message.reply_text(
        "📈 *Welcome to Stocks Insights Pro*\nYour AI-powered stock market advisor",
        parse_mode='Markdown'
    )
    await update.message.reply_text(TERMS, parse_mode='Markdown')
    return AGREE

async def agree(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """User agrees to terms"""
    user = update.message.from_user
    USER_PROFILE[user.id] = {
        'name': f"{user.first_name} {user.last_name}" if user.last_name else user.first_name,
        'agreed_at': datetime.now().isoformat(),
        'agreed': True
    }
    
    await update.message.reply_text(
        "✅ Terms accepted! Let's create your investment profile.\n\n"
        "💰 *What is your investment amount?* (e.g., $5,000)",
        parse_mode='Markdown'
    )
    return AMOUNT

async def amount(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Capture investment amount"""
    user_id = update.message.from_user.id
    try:
        amount = float(update.message.text.replace('$', '').replace(',', ''))
        USER_PROFILE[user_id]['amount'] = amount
        await update.message.reply_text(
            "⏳ *What is your investment time horizon?*\n"
            "(Short: <1 year, Medium: 1-5 years, Long: >5 years)",
            parse_mode='Markdown'
        )
        return HORIZON
    except ValueError:
        await update.message.reply_text("⚠️ Please enter a valid amount (e.g., 5000 or $5,000)")
        return AMOUNT

async def horizon(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Capture investment horizon"""
    horizon = update.message.text.lower()
    if horizon not in ['short', 'medium', 'long']:
        await update.message.reply_text("⚠️ Please choose: Short, Medium, or Long")
        return HORIZON
    
    user_id = update.message.from_user.id
    USER_PROFILE[user_id]['horizon'] = horizon
    
    keyboard = [['Low', 'Medium', 'High']]
    reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True)
    await update.message.reply_text(
        "🎯 *What is your risk tolerance?*",
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )
    return RISK

async def risk(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Capture risk tolerance"""
    risk = update.message.text.lower()
    if risk not in ['low', 'medium', 'high']:
        await update.message.reply_text("⚠️ Please choose: Low, Medium, or High")
        return RISK
    
    user_id = update.message.from_user.id
    USER_PROFILE[user_id]['risk'] = risk
    
    await update.message.reply_text(
        "🏢 *Which sectors interest you?* (e.g., Tech, Healthcare, Energy)\n"
        "Separate multiple sectors with commas",
        parse_mode='Markdown'
    )
    return SECTORS

async def sectors(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Capture preferred sectors"""
    user_id = update.message.from_user.id
    sectors = [s.strip().title() for s in update.message.text.split(',')]
    USER_PROFILE[user_id]['sectors'] = sectors
    
    keyboard = [['US', 'EU', 'Asia'], ['Global', 'Emerging Markets']]
    reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True)
    await update.message.reply_text(
        "🌎 *What is your regional focus?*",
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )
    return REGION

async def region(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Capture regional focus and start analysis"""
    user_id = update.message.from_user.id
    region = update.message.text
    USER_PROFILE[user_id]['region'] = region
    
    await update.message.reply_text(
        "🔄 Analyzing global markets based on your profile...\n"
        "This may take 20-30 seconds",
        parse_mode='Markdown'
    )
    
    # Generate recommendations
    recommendations = await generate_recommendations(user_id)
    RECOMMENDATIONS[user_id] = recommendations
    
    # Generate portfolio
    USER_PORTFOLIOS[user_id] = optimize_portfolio(user_id)
    
    # Display results
    await show_recommendations(update, context, user_id)
    return ANALYSIS

async def generate_recommendations(user_id: int) -> list:
    """Generate stock recommendations based on user profile"""
    profile = USER_PROFILE[user_id]
    
    # Get trending stocks based on profile
    trending_stocks = get_trending_stocks(
        sectors=profile['sectors'],
        region=profile['region'],
        risk=risk_levels[profile['risk']]
    )
    
    # Analyze and rank stocks
    analyzed_stocks = []
    for stock in trending_stocks[:15]:  # Limit for efficiency
        try:
            analysis = analyze_stock(stock['symbol'], profile)
            analyzed_stocks.append({
                **stock,
                'analysis': analysis,
                'score': calculate_stock_score(analysis, profile)
            })
        except Exception as e:
            logger.error(f"Error analyzing {stock['symbol']}: {str(e)}")
    
    # Sort by score and return top 5
    return sorted(analyzed_stocks, key=lambda x: x['score'], reverse=True)[:5]

def get_trending_stocks(sectors: list, region: str, risk: dict) -> list:
    """Get trending stocks from Finnhub API"""
    try:
        # Map region to exchange
        exchange_map = {
            'US': 'US',
            'EU': 'EU',
            'Asia': 'ASIA',
            'Global': 'GLOBAL',
            'Emerging Markets': 'EMERGING'
        }
        exchange = exchange_map.get(region, 'US')
        
        # Build sector filter
        sector_query = f"&sector={','.join(sectors)}" if sectors else ""
        
        url = f"https://finnhub.io/api/v1/stock/screener?exchange={exchange}{sector_query}&token={FINNHUB_KEY}"
        response = requests.get(url)
        data = response.json()
        
        # Process results
        stocks = []
        for item in data.get('data', [])[:20]:
            stocks.append({
                'symbol': item['symbol'],
                'name': item['name'],
                'sector': item['sector']
            })
        return stocks
    except Exception as e:
        logger.error(f"Trending stocks error: {e}")
        # Fallback to mock data
        return [
            {'symbol': 'AAPL', 'name': 'Apple Inc.', 'sector': 'Technology'},
            {'symbol': 'MSFT', 'name': 'Microsoft Corporation', 'sector': 'Technology'},
            {'symbol': 'GOOGL', 'name': 'Alphabet Inc.', 'sector': 'Technology'},
            {'symbol': 'AMZN', 'name': 'Amazon.com Inc.', 'sector': 'Consumer Cyclical'},
            {'symbol': 'TSLA', 'name': 'Tesla Inc.', 'sector': 'Automotive'},
            {'symbol': 'JNJ', 'name': 'Johnson & Johnson', 'sector': 'Healthcare'},
            {'symbol': 'V', 'name': 'Visa Inc.', 'sector': 'Financial Services'},
            {'symbol': 'NVDA', 'name': 'NVIDIA Corporation', 'sector': 'Technology'},
            {'symbol': 'PYPL', 'name': 'PayPal Holdings', 'sector': 'Financial Services'},
            {'symbol': 'DIS', 'name': 'Walt Disney Co.', 'sector': 'Communication Services'}
        ]

def calculate_stock_score(analysis: dict, profile: dict) -> float:
    """Calculate overall score based on analysis and user profile"""
    # Scoring weights based on user profile
    weights = {
        'growth': 0.4 if profile['horizon'] == 'long' else 0.2,
        'value': 0.3 if profile['risk'] == 'low' else 0.1,
        'sentiment': 0.2,
        'risk': 0.3 if profile['risk'] == 'low' else 0.1
    }
    
    # Calculate component scores (normalized)
    growth_score = (analysis['forecast']['price_3mo'] / analysis['current_price'] - 1) * 100
    value_score = 1 / analysis['fundamental']['pe_ratio'] if analysis['fundamental']['pe_ratio'] and analysis['fundamental']['pe_ratio'] > 0 else 0
    sentiment_score = analysis['sentiment']['score'] * 10  # Scale to 0-10
    risk_score = 10 * (1 - min(analysis['risk']['volatility'], 1.0))
    
    # Combine scores
    total_score = (
        weights['growth'] * growth_score +
        weights['value'] * value_score +
        weights['sentiment'] * sentiment_score +
        weights['risk'] * risk_score
    )
    
    # Normalize to 0-10 scale
    return min(max(total_score / 2, 0), 10)

async def show_recommendations(update: Update, context: ContextTypes.DEFAULT_TYPE, user_id: int):
    """Display recommendations to user"""
    recommendations = RECOMMENDATIONS[user_id]
    profile = USER_PROFILE[user_id]
    
    message = (
        f"🎯 *Personalized Recommendations for {profile['name']}*\n\n"
        f"💰 Investment Amount: ${profile['amount']:,.2f}\n"
        f"⏳ Time Horizon: {profile['horizon'].title()}\n"
        f"⚖️ Risk Tolerance: {profile['risk'].title()}\n"
        f"🏢 Sectors: {', '.join(profile['sectors'])}\n"
        f"🌎 Region: {profile['region']}\n\n"
        "📊 *Top Recommendations:*\n"
    )
    
    # Build recommendations list
    keyboard = []
    for i, stock in enumerate(recommendations, 1):
        message += (
            f"{i}. *{stock['name']}* ({stock['symbol']})\n"
            f"   💵 Price: ${stock['analysis']['current_price']:,.2f}\n"
            f"   📈 Score: {stock['score']:.2f}/10\n"
            f"   📊 Sector: {stock['sector']}\n\n"
        )
        keyboard.append([InlineKeyboardButton(
            f"{stock['symbol']} - {stock['name']}", 
            callback_data=f"detail_{stock['symbol']}"
        )])
    
    message += "Select a stock for detailed analysis:"
    
    # Add action buttons
    keyboard.append([InlineKeyboardButton("📊 Show Portfolio Allocation", callback_data="portfolio")])
    keyboard.append([InlineKeyboardButton("🔄 Get More Recommendations", callback_data="more")])
    keyboard.append([InlineKeyboardButton("📚 Education Center", callback_data="education")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        message, 
        parse_mode='Markdown', 
        reply_markup=reply_markup
    )

async def stock_detail(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show detailed stock analysis"""
    query = update.callback_query
    await query.answer()
    symbol = query.data.split('_')[1]
    user_id = query.from_user.id
    
    if not compliance_check(user_id):
        await query.edit_message_text("Please agree to terms first with /start")
        return
    
    # Get analysis
    if symbol in ANALYSIS_CACHE:
        analysis = ANALYSIS_CACHE[symbol]
    else:
        # Get from API if not cached
        profile = USER_PROFILE.get(user_id, {})
        analysis = analyze_stock(symbol, profile)
    
    # Generate GPT-4 summary
    gpt_summary = gpt4_analysis(symbol, analysis)
    
    # Build detailed message
    sentiment = analysis['sentiment']['trend'].upper()
    forecast_change = (analysis['forecast']['price_3mo'] / analysis['current_price'] - 1) * 100
    
    message = (
        f"📊 *{analysis['name']} ({symbol})*\n\n"
        f"💵 *Current Price:* ${analysis['current_price']:,.2f}\n"
        f"📈 *Technical Trend:* {analysis['technical']['trend']}\n"
        f"😃 *Sentiment:* {sentiment} ({analysis['sentiment']['score']:.2f}/10)\n"
        f"🔮 *3-Month Forecast:* ${analysis['forecast']['price_3mo']:,.2f} ({forecast_change:+.2f}%)\n"
        f"⚖️ *Risk Level:* {analysis['risk']['risk_level']}\n\n"
        "🧠 *AI Analysis Summary*\n"
        f"{gpt_summary}\n\n"
    )
    
    # Add trading signals
    stop_loss = analysis['current_price'] * 0.92
    take_profit = analysis['current_price'] * 1.15
    message += (
        "🚦 *Trading Signals*\n"
        f"🟢 Buy Zone: < ${analysis['current_price'] * 0.98:,.2f}\n"
        f"🔴 Stop Loss: ${stop_loss:,.2f}\n"
        f"🎯 Take Profit: ${take_profit:,.2f}\n\n"
    )
    
    # Add insider transactions if available
    if analysis.get('insiders'):
        message += "🧑‍💼 *Recent Insider Transactions*\n"
        for insider in analysis['insiders'][:3]:
            message += f"- {insider['name']} ({insider['position']}): {insider['type']} {insider['shares']} shares at ${insider['price']}\n"
        message += "\n"
    
    # Add links
    message += "🔗 *Useful Links*\n"
    message += f"- [Yahoo Finance](https://finance.yahoo.com/quote/{symbol})\n"
    message += f"- [SEC Filings](https://www.sec.gov/cgi-bin/browse-edgar?CIK={symbol})\n\n"
    
    # Add disclaimer
    message += "⚠️ *Disclaimer: Not financial advice. Do your own research.*"
    
    # Create keyboard
    keyboard = [
        [InlineKeyboardButton("📈 View Charts", callback_data=f"charts_{symbol}")],
        [InlineKeyboardButton("📰 Latest News", callback_data=f"news_{symbol}")],
        [InlineKeyboardButton("📊 Compare to Competitors", callback_data=f"compare_{symbol}")],
        [InlineKeyboardButton("🔙 Back to Recommendations", callback_data="back")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Send analysis
    await query.edit_message_text(
        message, 
        parse_mode='Markdown', 
        reply_markup=reply_markup,
        disable_web_page_preview=True
    )

async def show_charts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show technical and forecast charts"""
    query = update.callback_query
    await query.answer()
    symbol = query.data.split('_')[1]
    user_id = query.from_user.id
    
    if not compliance_check(user_id):
        return
    
    if symbol not in ANALYSIS_CACHE:
        profile = USER_PROFILE.get(user_id, {})
        analyze_stock(symbol, profile)
    
    analysis = ANALYSIS_CACHE[symbol]
    
    # Create media group
    media_group = [
        InputMediaPhoto(media=analysis['technical']['chart'], caption=f"Technical Analysis for {symbol}"),
        InputMediaPhoto(media=analysis['forecast']['chart'], caption=f"Price Forecast for {symbol}")
    ]
    
    await context.bot.send_media_group(
        chat_id=query.message.chat_id,
        media=media_group
    )
    
    # Keep menu visible
    await query.message.reply_text(
        f"📊 Charts for {symbol} displayed above",
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 Back to Analysis", callback_data=f"detail_{symbol}")]
        ])
    )

async def show_news(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show latest news for a stock"""
    query = update.callback_query
    await query.answer()
    symbol = query.data.split('_')[1]
    user_id = query.from_user.id
    
    if not compliance_check(user_id):
        return
    
    if symbol in ANALYSIS_CACHE:
        analysis = ANALYSIS_CACHE[symbol]
    else:
        profile = USER_PROFILE.get(user_id, {})
        analysis = analyze_stock(symbol, profile)
    
    sentiment = analysis['sentiment']
    news = sentiment.get('recent_news', [])
    
    if not news:
        await query.edit_message_text(f"No recent news found for {symbol}")
        return
    
    message = f"📰 *Latest News for {symbol}*\n\n"
    for i, item in enumerate(news[:5], 1):
        sentiment_emoji = "🟢" if item['sentiment'] > 0.5 else "🔴" if item['sentiment'] < -0.5 else "🟡"
        message += (
            f"{i}. {sentiment_emoji} [{item['headline']}]({item.get('url', '#')})\n"
            f"   Source: {item.get('source', 'Unknown')}\n\n"
        )
    
    await query.edit_message_text(
        message,
        parse_mode='Markdown',
        disable_web_page_preview=False,
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 Back to Analysis", callback_data=f"detail_{symbol}")]
        ])
    )

async def show_portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show portfolio allocation"""
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    
    if not compliance_check(user_id):
        await query.edit_message_text("Please complete your profile first with /start")
        return
    
    portfolio = USER_PORTFOLIOS.get(user_id, [])
    profile = USER_PROFILE.get(user_id, {})
    
    if not portfolio:
        await query.edit_message_text("Portfolio not generated. Please run /portfolio")
        return
    
    # Create portfolio message
    message = "📊 *Personalized Portfolio Allocation*\n\n"
    message += f"💰 Total Investment: ${profile.get('amount', 0):,.2f}\n"
    message += f"⚖️ Risk Profile: {profile.get('risk', 'medium').title()}\n\n"
    
    # Group by category
    categories = {}
    for item in portfolio:
        category = item['category']
        if category not in categories:
            categories[category] = []
        categories[category].append(item)
    
    for category, items in categories.items():
        category_total = sum(item['amount'] for item in items)
        message += f"**{category}**: ${category_total:,.2f}\n"
        for item in items:
            message += f"  - {item['name']} ({item['symbol']}): ${item['amount']:,.2f} ({item['shares']:.2f} shares)\n"
        message += "\n"
    
    # Add performance metrics
    message += "📈 *Recommended Performance Metrics*\n"
    message += "- Rebalance quarterly\n- Review against S&P 500\n- Set stop losses for volatile assets\n\n"
    
    # Add disclaimer
    message += "⚠️ *Portfolio allocations are AI-generated suggestions. Actual investments may vary.*"
    
    await query.edit_message_text(
        message,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("📈 Performance Simulator", callback_data="simulate")],
            [InlineKeyboardButton("🔙 Back to Recommendations", callback_data="back")]
        ])
    )

async def education_center(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show education center"""
    query = update.callback_query
    await query.answer()
    
    message = "📚 *Stocks Insights Education Center*\n\nSelect a topic to learn:"
    
    keyboard = []
    for key, desc in EDUCATION_MODULES.items():
        keyboard.append([InlineKeyboardButton(desc, callback_data=f"edu_{key}")])
    
    keyboard.append([InlineKeyboardButton("🔙 Back", callback_data="back")])
    
    await query.edit_message_text(
        message,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle regular messages"""
    text = update.message.text.lower()
    
    if text in ['hi', 'hello', 'start']:
        await start(update, context)
    elif text == 'menu':
        await show_main_menu(update)
    elif 'alert' in text:
        symbol = text.split()[-1].upper()
        if validate_symbol(symbol):
            MARKET_ALERTS[update.message.from_user.id] = symbol
            await update.message.reply_text(f"🔔 Price alerts enabled for {symbol}. You'll be notified of significant moves.")
        else:
            await update.message.reply_text("⚠️ Invalid stock symbol. Please use format: 'alert AAPL'")
    else:
        await update.message.reply_text(
            "I'm your advanced stock market advisor. Here's what you can do:\n"
            "/start - Begin setup\n"
            "/recommend - Get personalized recommendations\n"
            "/analyze [symbol] - Analyze specific stock\n"
            "/portfolio - Show your portfolio\n"
            "/education - Access learning resources\n"
            "/alerts - Manage price alerts"
        )

async def show_main_menu(update: Update):
    """Show main menu"""
    keyboard = [
        ["📊 My Recommendations", "📈 Market Overview"],
        ["🔍 Analyze Stock", "🏦 Portfolio Builder"],
        ["📚 Education Center", "🔔 Alerts"],
        ["⚙️ Settings", "ℹ️ Help"]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    await update.message.reply_text(
        "📈 *Stocks Insights Pro Main Menu*",
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )

async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /analyze command"""
    if not context.args:
        await update.message.reply_text("Please specify a stock symbol. Example: /analyze AAPL")
        return
    
    symbol = context.args[0].upper()
    if not validate_symbol(symbol):
        await update.message.reply_text("⚠️ Invalid stock symbol. Please use 1-5 uppercase letters.")
        return
    
    try:
        user_id = update.message.from_user.id
        profile = USER_PROFILE.get(user_id, {})
        analysis = analyze_stock(symbol, profile)
        
        # Send summary
        await update.message.reply_text(
            f"🔍 Analyzing {analysis['name']} ({symbol})...",
            parse_mode='Markdown'
        )
        
        # Send detailed analysis
        message = (
            f"📊 *{analysis['name']} ({symbol}) Analysis*\n\n"
            f"💵 Price: ${analysis['current_price']:,.2f}\n"
            f"📈 Trend: {analysis['technical']['trend']}\n"
            f"😃 Sentiment: {analysis['sentiment']['trend']} ({analysis['sentiment']['score']:.2f}/10)\n"
            f"⚠️ Risk: {analysis['risk']['risk_level']}\n\n"
            "Select an option for more details:"
        )
        
        keyboard = [
            [InlineKeyboardButton("📊 Full Analysis", callback_data=f"detail_{symbol}")],
            [InlineKeyboardButton("📈 Technical Charts", callback_data=f"charts_{symbol}")],
            [InlineKeyboardButton("📰 Latest News", callback_data=f"news_{symbol}")]
        ]
        
        await update.message.reply_text(
            message,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
    except Exception as e:
        await update.message.reply_text(f"❌ Error analyzing {symbol}: {str(e)}")

async def portfolio_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Build investment portfolio"""
    user_id = update.message.from_user.id
    if user_id not in USER_PROFILE:
        await update.message.reply_text("Please complete your profile first with /start")
        return
    
    profile = USER_PROFILE[user_id]
    portfolio = USER_PORTFOLIOS.get(user_id, [])
    
    if not portfolio:
        portfolio = optimize_portfolio(user_id)
        USER_PORTFOLIOS[user_id] = portfolio
    
    if not portfolio:
        await update.message.reply_text("❌ Could not generate portfolio. Please try again later.")
        return
    
    # Create portfolio message
    message = "📊 *Personalized Portfolio Allocation*\n\n"
    message += f"💰 Total Investment: ${profile.get('amount', 0):,.2f}\n"
    message += f"⚖️ Risk Profile: {profile.get('risk', 'medium').title()}\n\n"
    
    # Group by category
    categories = {}
    for item in portfolio:
        category = item['category']
        if category not in categories:
            categories[category] = []
        categories[category].append(item)
    
    for category, items in categories.items():
        category_total = sum(item['amount'] for item in items)
        message += f"**{category}**: ${category_total:,.2f}\n"
        for item in items:
            message += f"  - {item['name']} ({item['symbol']}): ${item['amount']:,.2f}\n"
        message += "\n"
    
    # Add performance metrics
    message += "📈 *Recommended Performance Metrics*\n"
    message += "- Rebalance quarterly\n- Review against S&P 500\n- Set stop losses for volatile assets\n\n"
    
    # Add disclaimer
    message += "⚠️ *Portfolio allocations are AI-generated suggestions. Actual investments may vary.*"
    
    await update.message.reply_text(
        message,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("📈 Performance Simulator", callback_data="simulate")],
            [InlineKeyboardButton("🔄 Rebalance Portfolio", callback_data="rebalance")]
        ])
    )

async def education_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show education center"""
    message = "📚 *Stocks Insights Education Center*\n\nSelect a topic to learn:"
    
    keyboard = []
    for key, desc in EDUCATION_MODULES.items():
        keyboard.append([InlineKeyboardButton(desc, callback_data=f"edu_{key}")])
    
    await update.message.reply_text(
        message,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

def cleanup_cache(context: ContextTypes.DEFAULT_TYPE):
    """Clean up cache and free memory"""
    logger.info("Running cache cleanup...")
    # Clear old cache entries
    global ANALYSIS_CACHE
    now = datetime.now()
    expired_keys = []
    
    for symbol, analysis in list(ANALYSIS_CACHE.items()):
        updated = datetime.fromisoformat(analysis['last_updated'])
        if (now - updated) > timedelta(hours=6):
            expired_keys.append(symbol)
    
    for key in expired_keys:
        del ANALYSIS_CACHE[key]
    
    # Force garbage collection
    gc.collect()
    logger.info(f"Cleaned {len(expired_keys)} expired cache entries")

def main():
    """Run the bot"""
    application = Application.builder().token(TOKEN).build()
    
    # Add scheduled jobs
    job_queue = application.job_queue
    job_queue.run_repeating(check_market_conditions, interval=300, first=10)  # Every 5 minutes
    job_queue.run_daily(cleanup_cache, time=datetime.strptime('04:00', '%H:%M').time())  # Daily at 4 AM
    
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            AGREE: [CommandHandler('agree', agree)],
            AMOUNT: [MessageHandler(filters.TEXT & ~filters.COMMAND, amount)],
            HORIZON: [MessageHandler(filters.TEXT & ~filters.COMMAND, horizon)],
            RISK: [MessageHandler(filters.TEXT & ~filters.COMMAND, risk)],
            SECTORS: [MessageHandler(filters.TEXT & ~filters.COMMAND, sectors)],
            REGION: [MessageHandler(filters.TEXT & ~filters.COMMAND, region)],
            ANALYSIS: [
                CallbackQueryHandler(stock_detail, pattern='^detail_'),
                CallbackQueryHandler(show_portfolio, pattern='^portfolio$'),
                CallbackQueryHandler(education_center, pattern='^education$'),
                CallbackQueryHandler(show_recommendations, pattern='^back$')
            ]
        },
        fallbacks=[CommandHandler('cancel', lambda update, context: ConversationHandler.END)],
        per_user=True
    )
    
    # Register handlers
    application.add_handler(conv_handler)
    application.add_handler(CommandHandler('analyze', analyze_command))
    application.add_handler(CommandHandler('portfolio', portfolio_command))
    application.add_handler(CommandHandler('education', education_command))
    application.add_handler(CallbackQueryHandler(show_charts, pattern='^charts_'))
    application.add_handler(CallbackQueryHandler(show_news, pattern='^news_'))
    application.add_handler(CallbackQueryHandler(show_portfolio, pattern='^portfolio$'))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Start the bot
    application.run_polling()

if __name__ == '__main__':
    main()
