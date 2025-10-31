import os
import time
from datetime import datetime, timedelta
from functools import wraps

import numpy as np
import pandas as pd
import yfinance as yf
from dateutil import parser
from flask import Flask, request, jsonify
from flask_cors import CORS

# Optional: requests for NewsAPI call
import requests

# -----------------------
# Simple in-memory cache
# -----------------------
class SimpleCache:
    def __init__(self):
        self.store = {}  # key => (expires_at_unix, value)

    def set(self, key, value, ttl_seconds=60):
        self.store[key] = (time.time() + ttl_seconds, value)

    def get(self, key):
        pair = self.store.get(key)
        if not pair:
            return None
        expires_at, val = pair
        if time.time() > expires_at:
            del self.store[key]
            return None
        return val

cache = SimpleCache()

# -----------------------
# App setup
# -----------------------
app = Flask(__name__)
CORS(app)

# Environment variables
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY', '')  # optional; if absent, server returns demo news

# -----------------------
# Utilities
# -----------------------
def cached(ttl=60):
    """Decorator to cache endpoint function results in memory (simple)."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            key = f"{fn.__name__}|" + "|".join(map(str, args)) + "|" + "|".join(f"{k}={v}" for k, v in kwargs.items())
            val = cache.get(key)
            if val is not None:
                return val
            val = fn(*args, **kwargs)
            cache.set(key, val, ttl_seconds=ttl)
            return val
        return wrapper
    return decorator

def get_full_ticker(ticker):
    """Append .NS for common Indian tickers if user passed plain symbol (keeps behavior from your old helper)."""
    if not ticker:
        return ticker
    t = ticker.strip().upper()
    if '.' not in t and any(x in t for x in ['RELIANCE', 'TCS', 'HDFC', 'INFY', 'ICICI', 'HINDUNILVR']):
        return f"{t}.NS"
    return t

def to_date_strings(index):
    return [d.strftime('%Y-%m-%d') for d in pd.to_datetime(index)]

# Technical indicators (simple)
def sma(series, period):
    return series.rolling(period).mean()

def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, 1e-8))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal
    return macd_line, signal, hist

def linear_trend_forecast(dates, prices, days_ahead=7, fit_window=30):
    """
    Simple linear-trend forecast:
    - uses last `fit_window` points to fit a linear model price ~ day_index
    - returns forecast_prices list of length days_ahead and R^2 as 'confidence'
    """
    if len(prices) == 0:
        return [], 0.0

    # convert dates to ordinal numbers for regression
    x = np.arange(len(prices))
    y = np.array(prices, dtype=float)

    # use last fit_window points
    if len(x) > fit_window:
        x_fit = x[-fit_window:]
        y_fit = y[-fit_window:]
        x0 = x[-1]
    else:
        x_fit = x
        y_fit = y
        x0 = x[-1]

    # linear regression via polyfit
    try:
        a, b = np.polyfit(x_fit, y_fit, deg=1)  # y = a*x + b
        y_pred_fit = a * x_fit + b
        # compute R^2
        ss_res = np.sum((y_fit - y_pred_fit) ** 2)
        ss_tot = np.sum((y_fit - y_fit.mean()) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-12))
    except Exception:
        a, b, r2 = 0.0, float(y[-1]), 0.0

    forecast_prices = []
    forecast_dates = []
    # next calendar dates (we keep calendar days; frontend can show them)
    last_date = None
    try:
        last_date = pd.to_datetime(dates[-1])
    except Exception:
        last_date = pd.Timestamp.now()

    for i in range(1, days_ahead + 1):
        xi = x0 + i
        yi = a * xi + b
        forecast_prices.append(float(yi))
        forecast_dates.append((last_date + timedelta(days=i)).strftime('%Y-%m-%d'))

    confidence_pct = max(0.0, min(100.0, float(r2 * 100)))  # clamp 0-100

    return list(zip(forecast_dates, forecast_prices)), confidence_pct

# -----------------------
# Endpoints
# -----------------------

@app.route('/stock_details', methods=['GET'])
def stock_details():
    ticker_symbol = request.args.get('ticker', '')
    if not ticker_symbol:
        return jsonify({"error": "Ticker symbol is required"}), 400

    full_ticker = get_full_ticker(ticker_symbol)
    cache_key = f"details::{full_ticker}"
    cached_val = cache.get(cache_key)
    if cached_val:
        return jsonify(cached_val)

    try:
        tk = yf.Ticker(full_ticker)
        info = tk.info or {}

        # Validate presence of price
        price = info.get('regularMarketPrice') or info.get('previousClose') or None
        if price is None:
            return jsonify({"error": f"No valid data found for ticker {ticker_symbol} ({full_ticker})"}), 404

        # Build response with useful fields (frontend expects some of these)
        details = {
            "symbol": info.get('symbol', full_ticker),
            "longName": info.get('longName') or info.get('shortName') or full_ticker,
            "exchange": info.get('exchange', 'NSE'),
            "currentPrice": float(price),
            "change": float((info.get('regularMarketChange') or 0.0)),
            "changePercent": float((info.get('regularMarketChangePercent') or 0.0)),
            "sector": info.get('sector') or 'N/A',
            "country": info.get('country') or 'India',
            "marketCap": info.get('marketCap') or None,
            "peRatio": info.get('trailingPE') or info.get('forwardPE') or None,
            "revenueTTM": info.get('totalRevenue') or info.get('revenue') or None,
            "longBusinessSummary": info.get('longBusinessSummary', '')[:1500],  # truncated summary
        }

        cache.set(cache_key, details, ttl_seconds=120)  # short TTL
        return jsonify(details)
    except Exception as e:
        return jsonify({"error": f"Error fetching stock details: {str(e)}"}), 500

@app.route('/predict', methods=['GET'])
def predict():
    """
    Returns:
    {
      "historical": [{ "Date": "YYYY-MM-DD", "Close": float }, ...],
      "forecast":  [{ "Date": "YYYY-MM-DD", "Close": float }, ...],
      "model": "LinearTrend",
      "confidence": "82.3",
      "indicators": { "ma7": [...], "ma30": [...], "rsi14": last_value_or_null, "macd": last_macd_value_or_null }
    }
    """
    ticker_symbol = request.args.get('ticker', '')
    if not ticker_symbol:
        return jsonify({"error": "Ticker symbol is required"}), 400

    range_param = request.args.get('range', '1y')  # currently used to fetch period from yfinance
    full_ticker = get_full_ticker(ticker_symbol)
    cache_key = f"predict::{full_ticker}::range={range_param}"
    cached_val = cache.get(cache_key)
    if cached_val:
        return jsonify(cached_val)

    try:
        tk = yf.Ticker(full_ticker)

        # Attempt to fetch historical data. Use 'period' derived from range_param
        # Accept values like '1m','3m','6m','1y','5y'
        allowed_periods = {'1m', '3m', '6m', '1y', '5y'}
        yf_period = range_param if range_param in allowed_periods else '1y'
        hist_df = tk.history(period=yf_period, auto_adjust=False)

        # If empty, try fallback to 1y
        if hist_df.empty:
            hist_df = tk.history(period='1y')

        if hist_df.empty:
            return jsonify({"error": f"No historical data found for ticker {full_ticker}."}), 404

        # Keep only Date index and Close column
        hist_df = hist_df[['Close']].copy()
        hist_df.reset_index(inplace=True)  # Date as column
        hist_df['Date'] = pd.to_datetime(hist_df['Date']).dt.date
        hist_df['Close'] = hist_df['Close'].astype(float)

        # Indicators
        price_series = hist_df['Close']
        ma7 = sma(price_series, 7).tolist()
        ma30 = sma(price_series, 30).tolist()
        rsi_series = compute_rsi(price_series, period=14)
        rsi_last = float(rsi_series.dropna().iloc[-1]) if not rsi_series.dropna().empty else None
        macd_line, macd_signal, macd_hist = compute_macd(price_series)
        macd_last = float(macd_line.dropna().iloc[-1]) if not macd_line.dropna().empty else None

        # Forecast (simple linear trend using last 60 trading points by default)
        dates_list = [d.strftime('%Y-%m-%d') for d in hist_df['Date'].to_list()]
        closes_list = hist_df['Close'].to_list()

        forecast_pairs, confidence_pct = linear_trend_forecast(dates_list, closes_list, days_ahead=7, fit_window=60)
        forecast_df = [{"Date": d, "Close": float(p)} for d, p in forecast_pairs]

        # Prepare historical for JSON
        historical_json = [{"Date": d.strftime('%Y-%m-%d'), "Close": float(c)} for d, c in zip(hist_df['Date'], hist_df['Close'])]

        result = {
            "historical": historical_json,
            "forecast": forecast_df,
            "model": "LinearTrend (polyfit deg=1)",
            "confidence": round(float(confidence_pct), 2),
            "indicators": {
                "ma7": [None if pd.isna(x) else float(x) for x in ma7],
                "ma30": [None if pd.isna(x) else float(x) for x in ma30],
                "rsi14_last": rsi_last,
                "macd_last": macd_last
            }
        }

        cache.set(cache_key, result, ttl_seconds=90)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

@app.route('/news', methods=['GET'])
def news():
    """
    Returns top news articles for the ticker/company.
    Query: ?ticker=RELIANCE.NS&limit=5
    If NEWSAPI_KEY is not set, returns demo articles.
    """
    ticker = request.args.get('ticker', '')
    if not ticker:
        return jsonify({"error": "ticker is required"}), 400
    limit = int(request.args.get('limit', 5))
    cache_key = f"news::{ticker}::limit={limit}"
    cached_val = cache.get(cache_key)
    if cached_val:
        return jsonify(cached_val)

    company_query = ticker.replace('.NS', '').replace('.BO', '')

    if not NEWSAPI_KEY:
        # Demo fallback
        demo = {
            "source": "demo",
            "articles": [
                {"title": f"{company_query} announces strategic partnership", "source": {"name": "Economic Times"}, "url": ""},
                {"title": f"Analyst upgrades {company_query} to Buy", "source": {"name": "Mint"}, "url": ""},
                {"title": f"{company_query} quarterly results beat estimates", "source": {"name": "Business Standard"}, "url": ""},
            ][:limit]
        }
        cache.set(cache_key, demo, ttl_seconds=300)
        return jsonify(demo)

    # Call NewsAPI (example)
    try:
        q = company_query
        url = "https://newsapi.org/v2/everything"
        params = {"q": q, "pageSize": limit, "apiKey": NEWSAPI_KEY, "language": "en", "sortBy": "relevancy"}
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        cache.set(cache_key, data, ttl_seconds=300)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": f"News fetch failed: {str(e)}", "articles": []}), 500

@app.route('/compare', methods=['GET'])
def compare():
    """
    Returns aligned historical series for t1 and t2 so frontend can overlay them.
    Query: ?t1=RELIANCE.NS&t2=TCS.NS&range=1y
    """
    t1 = request.args.get('t1', '')
    t2 = request.args.get('t2', '')
    rng = request.args.get('range', '1y')
    if not t1 or not t2:
        return jsonify({"error": "t1 and t2 are required"}), 400

    full1 = get_full_ticker(t1)
    full2 = get_full_ticker(t2)
    cache_key = f"compare::{full1}::{full2}::range={rng}"
    cached_val = cache.get(cache_key)
    if cached_val:
        return jsonify(cached_val)

    try:
        tk1 = yf.Ticker(full1)
        tk2 = yf.Ticker(full2)

        df1 = tk1.history(period=rng)[['Close']].rename(columns={'Close': 'Close1'}).reset_index()
        df2 = tk2.history(period=rng)[['Close']].rename(columns={'Close': 'Close2'}).reset_index()

        if df1.empty or df2.empty:
            return jsonify({"error": "Historical data missing for one or both tickers."}), 404

        # convert dates and align by Date (inner join)
        df1['Date'] = pd.to_datetime(df1['Date']).dt.date
        df2['Date'] = pd.to_datetime(df2['Date']).dt.date
        merged = pd.merge(df1[['Date', 'Close1']], df2[['Date', 'Close2']], on='Date', how='inner')
        merged = merged.sort_values('Date')
        result = {
            "dates": [d.strftime('%Y-%m-%d') for d in merged['Date'].tolist()],
            "series1": [float(x) for x in merged['Close1'].tolist()],
            "series2": [float(x) for x in merged['Close2'].tolist()],
            "symbols": [full1, full2]
        }
        cache.set(cache_key, result, ttl_seconds=120)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Compare failed: {str(e)}"}), 500

# -----------------------
# Run
# -----------------------
if __name__ == '__main__':
    # For local development:
    # 1) export NEWSAPI_KEY="your_key"   (optional)
    # 2) python app.py
    # For network testing, change host to '0.0.0.0'
    app.run(debug=True, host='127.0.0.1', port=5000)
