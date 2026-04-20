import os
import time
import requests
import feedparser
from datetime import datetime
from binance.client import Client
from dotenv import load_dotenv
import pandas as pd
import ta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()
client = Client(os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_SECRET_KEY"))
analyzer = SentimentIntensityAnalyzer()

# Keyword untuk analisis kualitatif
BULLISH_KEYWORDS = [
    'surge', 'rally', 'breakout', 'bullish', 'adoption', 'approval', 'partnership',
    'launch', 'upgrade', 'all-time high', 'ath', 'institutional', 'etf approved',
    'buy', 'pump', 'moon', 'accumulation', 'halving', 'positive', 'growth',
    'breakthrough', 'support', 'bullrun', 'green', 'profit'
]

BEARISH_KEYWORDS = [
    'crash', 'dump', 'bearish', 'ban', 'regulation', 'hack', 'scam', 'fraud',
    'lawsuit', 'sec', 'investigation', 'sell-off', 'correction', 'decline',
    'fall', 'drop', 'negative', 'warning', 'concern', 'fear', 'liquidation',
    'bear market', 'red', 'loss', 'bankruptcy', 'collapse', 'exploit'
]

HIGH_IMPACT_EVENTS = [
    'fed', 'fomc', 'powell', 'interest rate', 'inflation', 'cpi', 'ppi', 'gdp',
    'recession', 'war', 'bitcoin etf', 'spot etf', 'halving', 'sec approval',
    'china', 'trump', 'biden', 'treasury', 'sanctions', 'stimulus'
]

# ==================== BINANCE DATA ====================

def get_market_data(symbol="BTCUSDT", limit=200):
    klines = client.get_klines(symbol=symbol, interval="5m", limit=limit)
    df = pd.DataFrame(klines, columns=[
        "time","open","high","low","close","volume",
        "close_time","quote_vol","trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ])
    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
    return df

def calc_indicators(df):
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["rsi_fast"] = ta.momentum.RSIIndicator(df["close"], window=7).rsi()
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    bb = ta.volatility.BollingerBands(df["close"], window=20)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()
    stoch = ta.momentum.StochRSIIndicator(df["close"], window=14)
    df["stoch_k"] = stoch.stochrsi_k() * 100
    df["stoch_d"] = stoch.stochrsi_d() * 100
    df["ema_12"] = ta.trend.EMAIndicator(df["close"], window=12).ema_indicator()
    df["ema_26"] = ta.trend.EMAIndicator(df["close"], window=26).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df["ema_200"] = ta.trend.EMAIndicator(df["close"], window=200).ema_indicator()
    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    df["adx"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14).adx()
    df["volume_sma"] = df["volume"].rolling(window=20).mean()
    return df.dropna()

# ==================== MARKET SENTIMENT ====================

def get_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        d = r.json()["data"][0]
        return {"value": int(d["value"]), "class": d["value_classification"]}
    except:
        return {"value": 50, "class": "Unknown"}

def get_funding_rate(symbol="BTCUSDT"):
    try:
        r = requests.get(f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}", timeout=10)
        return float(r.json()["lastFundingRate"]) * 100
    except:
        return 0.0

def get_long_short_ratio(symbol="BTCUSDT"):
    try:
        r = requests.get(f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={symbol}&period=1h&limit=1", timeout=10)
        d = r.json()[0]
        return {"long": float(d["longAccount"])*100, "short": float(d["shortAccount"])*100, "ratio": float(d["longShortRatio"])}
    except:
        return {"long": 50, "short": 50, "ratio": 1.0}

def get_whale_trades(symbol="BTCUSDT", min_qty=5):
    try:
        r = requests.get(f"https://api.binance.com/api/v3/trades?symbol={symbol}&limit=500", timeout=10)
        trades = r.json()
        buys, sells = 0, 0
        for t in trades:
            qty = float(t["qty"])
            if qty >= min_qty:
                if t["isBuyerMaker"]:
                    sells += qty
                else:
                    buys += qty
        return {"buy": buys, "sell": sells, "pressure": "BUY" if buys > sells else "SELL"}
    except:
        return {"buy": 0, "sell": 0, "pressure": "NEUTRAL"}

# ==================== AI SENTIMENT ANALYSIS ====================

def analyze_news_sentiment(title):
    """Analisis sentiment berita dengan VADER + keyword detection"""
    scores = analyzer.polarity_scores(title)
    
    title_lower = title.lower()
    bullish_count = sum(1 for kw in BULLISH_KEYWORDS if kw in title_lower)
    bearish_count = sum(1 for kw in BEARISH_KEYWORDS if kw in title_lower)
    
    compound = scores['compound']
    keyword_bias = (bullish_count - bearish_count) * 0.1
    final_score = compound + keyword_bias
    
    if final_score > 0.3:
        sentiment = "BULLISH 🟢"
    elif final_score < -0.3:
        sentiment = "BEARISH 🔴"
    else:
        sentiment = "NEUTRAL ⚪"
    
    is_high_impact = any(event in title_lower for event in HIGH_IMPACT_EVENTS)
    impact = "HIGH ⚠️" if is_high_impact else "MEDIUM" if abs(final_score) > 0.5 else "LOW"
    
    return {
        "score": round(final_score, 3),
        "sentiment": sentiment,
        "impact": impact,
        "bullish_kw": bullish_count,
        "bearish_kw": bearish_count
    }

def get_crypto_news_analyzed():
    """Ambil berita dan analisis sentimen"""
    news_list = []
    
    sources = [
        ("https://cointelegraph.com/rss", "Cointelegraph"),
        ("https://cryptopotato.com/feed/", "CryptoPotato"),
        ("https://news.bitcoin.com/feed/", "Bitcoin.com"),
    ]
    
    for url, source_name in sources:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:4]:
                title = entry.get("title", "")[:150]
                sentiment_data = analyze_news_sentiment(title)
                news_list.append({
                    "title": title,
                    "source": source_name,
                    **sentiment_data
                })
        except:
            continue
    
    return news_list[:10]

def get_economic_events_analyzed():
    """Ambil economic events dan analisis impact"""
    events = []
    try:
        feed = feedparser.parse("https://www.investing.com/rss/news_285.rss")
        for entry in feed.entries[:5]:
            title = entry.title[:150]
            sentiment_data = analyze_news_sentiment(title)
            events.append({
                "title": title,
                "date": entry.get("published", "")[:16],
                **sentiment_data
            })
    except:
        pass
    return events

def calculate_aggregate_sentiment(news_list):
    """Hitung total sentimen dari semua berita"""
    if not news_list:
        return {"avg_score": 0, "bias": "NEUTRAL", "bullish_count": 0, "bearish_count": 0, "high_impact": 0}
    
    total_score = sum(n['score'] for n in news_list)
    avg_score = total_score / len(news_list)
    
    bullish = sum(1 for n in news_list if "BULLISH" in n['sentiment'])
    bearish = sum(1 for n in news_list if "BEARISH" in n['sentiment'])
    high_impact = sum(1 for n in news_list if "HIGH" in n['impact'])
    
    if avg_score > 0.2:
        bias = "STRONG BULLISH 🚀"
    elif avg_score > 0.05:
        bias = "BULLISH 📈"
    elif avg_score < -0.2:
        bias = "STRONG BEARISH 📉"
    elif avg_score < -0.05:
        bias = "BEARISH 🔻"
    else:
        bias = "NEUTRAL ➡️"
    
    return {
        "avg_score": round(avg_score, 3),
        "bias": bias,
        "bullish_count": bullish,
        "bearish_count": bearish,
        "high_impact": high_impact
    }

# ==================== GLOBAL MARKET ====================

def get_global_data():
    try:
        r = requests.get("https://api.coingecko.com/api/v3/global", timeout=10)
        d = r.json()["data"]
        return {
            "total_mcap": d["total_market_cap"]["usd"],
            "btc_dominance": d["market_cap_percentage"]["btc"],
            "mcap_change_24h": d["market_cap_change_percentage_24h_usd"]
        }
    except:
        return None

# ==================== MAIN ANALYSIS ====================

def analyze_complete(symbol="BTCUSDT"):
    print(f"\n🔄 Mengumpulkan & menganalisis data {symbol}...")
    
    df = get_market_data(symbol, limit=200)
    df = calc_indicators(df)
    latest = df.iloc[-1]
    
    fg = get_fear_greed()
    funding = get_funding_rate(symbol)
    ls = get_long_short_ratio(symbol)
    whale = get_whale_trades(symbol)
    global_data = get_global_data()
    
    print("🧠 Menganalisis sentimen berita...")
    news = get_crypto_news_analyzed()
    events = get_economic_events_analyzed()
    aggregate = calculate_aggregate_sentiment(news)
    
    support = df["low"].tail(50).min()
    resistance = df["high"].tail(50).max()
    trend_long = "BULLISH" if latest['ema_50'] > latest['ema_200'] else "BEARISH"
    trend_short = "BULLISH" if latest['ema_12'] > latest['ema_26'] else "BEARISH"
    
    print("\n" + "="*85)
    print(f"🎯 PRO TRADING BOT v4 - {symbol} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*85)
    print("\n📋 COPY SEMUA INI KE CLAUDE PRO:\n")
    
    output = f"""
Analisis LENGKAP Trading {symbol} dengan Quantitative + Qualitative Data:

========== 1. PRICE & TREND ==========
Harga Sekarang         : ${latest['close']:.2f}
Support / Resistance   : ${support:.2f} / ${resistance:.2f}
EMA 12 / 26            : ${latest['ema_12']:.2f} / ${latest['ema_26']:.2f}
EMA 50 / 200           : ${latest['ema_50']:.2f} / ${latest['ema_200']:.2f}
Trend Long / Short     : {trend_long} / {trend_short}

========== 2. MOMENTUM ==========
RSI 14 / 7             : {latest['rsi']:.2f} / {latest['rsi_fast']:.2f}
Stochastic K / D       : {latest['stoch_k']:.2f} / {latest['stoch_d']:.2f}
MACD / Signal / Hist   : {latest['macd']:.4f} / {latest['macd_signal']:.4f} / {latest['macd_hist']:.4f}
ADX                    : {latest['adx']:.2f} ({'STRONG' if latest['adx'] > 25 else 'WEAK'})

========== 3. VOLATILITY ==========
Bollinger U/M/L        : ${latest['bb_upper']:.2f} / ${latest['bb_mid']:.2f} / ${latest['bb_lower']:.2f}
ATR                    : ${latest['atr']:.2f}

========== 4. VOLUME ==========
Volume / MA            : {latest['volume']:,.2f} / {latest['volume_sma']:,.2f}
Status                 : {'TINGGI' if latest['volume'] > latest['volume_sma'] else 'RENDAH'}

========== 5. MARKET SENTIMENT ==========
Fear & Greed Index     : {fg['value']}/100 ({fg['class']})
Funding Rate           : {funding:.4f}%
Long/Short Ratio       : {ls['ratio']:.2f} (L:{ls['long']:.1f}% S:{ls['short']:.1f}%)
Whale Buy / Sell       : {whale['buy']:.2f} / {whale['sell']:.2f} ({whale['pressure']})
"""
    
    if global_data:
        output += f"""
========== 6. GLOBAL CRYPTO MARKET ==========
Total Market Cap       : ${global_data['total_mcap']/1e12:.2f}T
MCap Change 24h        : {global_data['mcap_change_24h']:.2f}%
BTC Dominance          : {global_data['btc_dominance']:.2f}%
"""
    
    output += f"""
========== 7. 🧠 AI NEWS SENTIMENT ANALYSIS ==========
Aggregate Bias         : {aggregate['bias']}
Average Sentiment      : {aggregate['avg_score']} (range: -1 to +1)
Bullish News           : {aggregate['bullish_count']} berita
Bearish News           : {aggregate['bearish_count']} berita
High Impact News       : {aggregate['high_impact']} berita
"""
    
    if news:
        output += "\n--- Detail Berita (sudah dianalisis AI) ---\n"
        for i, n in enumerate(news[:8], 1):
            output += f"\n{i}. [{n['source']}] {n['sentiment']} | Impact: {n['impact']} | Score: {n['score']}"
            output += f"\n   \"{n['title']}\""
            if n['bullish_kw'] > 0 or n['bearish_kw'] > 0:
                output += f"\n   Keywords: +{n['bullish_kw']} bullish, -{n['bearish_kw']} bearish"
    
    if events:
        output += "\n\n========== 8. 📅 ECONOMIC EVENTS (dengan Impact AI) ==========\n"
        for i, e in enumerate(events, 1):
            output += f"\n{i}. {e['sentiment']} | Impact: {e['impact']}"
            output += f"\n   \"{e['title']}\""
    
    output += """

=====================================================
Berikan analisis komprehensif & keputusan trading:

1. Signal: LONG / SHORT / WAIT
2. Confidence: 0-100%
3. Entry Price
4. Stop Loss
5. Take Profit (TP1, TP2, TP3)
6. Risk Level: LOW / MEDIUM / HIGH
7. Risk/Reward Ratio
8. Time Frame: SCALP / DAYTRADE / SWING
9. ⭐ Analisa kualitatif berita & event (impact ke harga)
10. ⭐ Konflik antara teknikal vs sentimen news? Mana yang dominan?
11. ⭐ Trigger events yang harus di-watch (Fed, CPI, ETF, dll)
12. Strategi backup jika market reversal
"""
    
    print(output)
    print("="*85)

# ==================== MAIN LOOP ====================

if __name__ == "__main__":
    print("🤖 PRO TRADING BOT v4.0 - WITH AI SENTIMENT ANALYSIS")
    print("📊 Data Sources:")
    print("   • Binance (Price, Volume, Futures, Whale)")
    print("   • Alternative.me (Fear & Greed)")
    print("   • CoinGecko (Global Market)")
    print("   • Multi-source News (CoinTelegraph, CryptoPotato, Bitcoin.com)")
    print("   • Investing.com (Economic Events)")
    print("🧠 AI Features:")
    print("   • VADER Sentiment Analysis")
    print("   • Keyword-based Bias Detection")
    print("   • Impact Level Classification")
    print("   • Aggregate Market Sentiment")
    print("\n⚠️  Pastikan VPN aktif!")
    print("Tekan Control + C untuk stop\n")
    
    while True:
        try:
            analyze_complete("BTCUSDT")
            print("\n⏳ Update otomatis 5 menit. Tekan Control+C untuk stop.")
            time.sleep(300)
        except KeyboardInterrupt:
            print("\n\n👋 Bot stopped.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(10)
