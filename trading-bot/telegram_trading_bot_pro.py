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
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
analyzer = SentimentIntensityAnalyzer()

BULLISH_KW = ['surge','rally','breakout','bullish','adoption','approval','partnership','upgrade','ath','etf','accumulation','halving']
BEARISH_KW = ['crash','dump','bearish','ban','regulation','hack','scam','sec','sell-off','liquidation','exploit']

# ==================== TELEGRAM ====================

def send_telegram(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        requests.post(url, data=data, timeout=10)
        return True
    except Exception as e:
        print(f"❌ Telegram error: {e}")
        return False

# ==================== DATA ====================

def get_klines(symbol, interval, limit=200):
    k = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(k, columns=["time","open","high","low","close","volume","close_time","quote_vol","trades","taker_buy_base","taker_buy_quote","ignore"])
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
    df["ema_9"] = ta.trend.EMAIndicator(df["close"], window=9).ema_indicator()
    df["ema_21"] = ta.trend.EMAIndicator(df["close"], window=21).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df["ema_200"] = ta.trend.EMAIndicator(df["close"], window=200).ema_indicator()
    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    df["adx"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14).adx()
    df["volume_sma"] = df["volume"].rolling(window=20).mean()
    return df.dropna()

def get_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        return int(r.json()["data"][0]["value"])
    except: return 50

def get_funding_rate(symbol):
    try:
        r = requests.get(f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}", timeout=10)
        return float(r.json()["lastFundingRate"]) * 100
    except: return 0.0

def get_long_short_ratio(symbol):
    try:
        r = requests.get(f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={symbol}&period=5m&limit=1", timeout=10)
        return float(r.json()[0]["longShortRatio"])
    except: return 1.0

def get_whale_pressure(symbol, min_qty=5):
    try:
        r = requests.get(f"https://api.binance.com/api/v3/trades?symbol={symbol}&limit=500", timeout=10)
        buys, sells = 0, 0
        for t in r.json():
            qty = float(t["qty"])
            if qty >= min_qty:
                if t["isBuyerMaker"]: sells += qty
                else: buys += qty
        return "🐳 BUY" if buys > sells * 1.2 else "🐳 SELL" if sells > buys * 1.2 else "NEUTRAL"
    except: return "NEUTRAL"

def get_news_sentiment():
    try:
        feed = feedparser.parse("https://cointelegraph.com/rss")
        scores = []
        for entry in feed.entries[:5]:
            title = entry.title.lower()
            score = analyzer.polarity_scores(title)['compound']
            bull = sum(1 for kw in BULLISH_KW if kw in title)
            bear = sum(1 for kw in BEARISH_KW if kw in title)
            final = score + (bull - bear) * 0.1
            scores.append(final)
        return sum(scores) / len(scores) if scores else 0
    except: return 0

# ==================== CONFLUENCE SCORING ====================

def calculate_confluence(df_5m, df_15m, df_1h, fear_greed, whale, sentiment):
    latest_5m = df_5m.iloc[-1]
    latest_1h = df_1h.iloc[-1]
    
    long_score = 0
    short_score = 0
    signals = []
    
    # 1. TREND
    if latest_1h['ema_50'] > latest_1h['ema_200']:
        long_score += 1
        signals.append("✓ 1H Bullish")
    else:
        short_score += 1
        signals.append("✓ 1H Bearish")
    
    # 2. RSI
    if latest_5m['rsi'] < 30:
        long_score += 1
        signals.append(f"✓ RSI Oversold ({latest_5m['rsi']:.0f})")
    elif latest_5m['rsi'] > 70:
        short_score += 1
        signals.append(f"✓ RSI Overbought ({latest_5m['rsi']:.0f})")
    
    # 3. MACD
    if latest_5m['macd'] > latest_5m['macd_signal']:
        long_score += 1
        signals.append("✓ MACD Bullish")
    else:
        short_score += 1
        signals.append("✓ MACD Bearish")
    
    # 4. STOCHASTIC
    if latest_5m['stoch_k'] < 20:
        long_score += 1
        signals.append(f"✓ Stoch Oversold ({latest_5m['stoch_k']:.0f})")
    elif latest_5m['stoch_k'] > 80:
        short_score += 1
        signals.append(f"✓ Stoch Overbought ({latest_5m['stoch_k']:.0f})")
    
    # 5. VOLUME
    if latest_5m['volume'] > latest_5m['volume_sma'] * 1.5:
        if latest_5m['close'] > latest_5m['open']:
            long_score += 1
            signals.append("✓ High Vol + Green")
        else:
            short_score += 1
            signals.append("✓ High Vol + Red")
    
    # 6. BB
    if latest_5m['close'] < latest_5m['bb_lower']:
        long_score += 1
        signals.append("✓ Below BB Lower")
    elif latest_5m['close'] > latest_5m['bb_upper']:
        short_score += 1
        signals.append("✓ Above BB Upper")
    
    # 7. WHALE
    if "BUY" in whale:
        long_score += 1
        signals.append("✓ Whale Buying")
    elif "SELL" in whale:
        short_score += 1
        signals.append("✓ Whale Selling")
    
    # 8. SENTIMENT
    if fear_greed < 25 and sentiment > 0.1:
        long_score += 1
        signals.append(f"✓ Extreme Fear + Bullish")
    elif fear_greed > 75 and sentiment < -0.1:
        short_score += 1
        signals.append(f"✓ Extreme Greed + Bearish")
    
    return long_score, short_score, signals

# ==================== ANALYSIS ====================

def analyze_and_send(symbol="BTCUSDT"):
    df_5m = calc_indicators(get_klines(symbol, "5m", 200))
    df_15m = calc_indicators(get_klines(symbol, "15m", 200))
    df_1h = calc_indicators(get_klines(symbol, "1h", 200))
    
    latest_5m = df_5m.iloc[-1]
    latest_1h = df_1h.iloc[-1]
    atr = latest_5m['atr']
    
    fg = get_fear_greed()
    funding = get_funding_rate(symbol)
    ls = get_long_short_ratio(symbol)
    whale = get_whale_pressure(symbol)
    sentiment = get_news_sentiment()
    
    long_score, short_score, signals = calculate_confluence(df_5m, df_15m, df_1h, fg, whale, sentiment)
    
    entry = latest_5m['close']
    sl_long = entry - (atr * 1.5)
    tp1_long = entry + (atr * 2.0)
    tp2_long = entry + (atr * 4.5)
    rr_long = (tp2_long - entry) / (entry - sl_long) if entry > sl_long else 0
    
    sl_short = entry + (atr * 1.5)
    tp1_short = entry - (atr * 2.0)
    tp2_short = entry - (atr * 4.5)
    rr_short = (entry - tp2_short) / (sl_short - entry) if sl_short > entry else 0
    
    message = f"""🎯 *5-MIN SIGNAL {symbol}*
📅 {datetime.now().strftime('%H:%M:%S')}

💰 *Price*: ${latest_5m['close']:,.2f}
📊 *Confluence*: L:{long_score}/8 vs S:{short_score}/8
📈 *ADX*: {latest_5m['adx']:.0f} ({'TREND' if latest_5m['adx'] > 20 else 'RANGE'})
🐋 {whale}

━━━━━━━━━━━━━━━━━━━

📌 *Signals:*
{chr(10).join(signals[:5])}

━━━━━━━━━━━━━━━━━━━

🟢 *LONG* (Confluence: {long_score}/8):
Entry: `${entry:,.2f}`
SL: `${sl_long:,.2f}`
TP1: `${tp1_long:,.2f}`
TP2: `${tp2_long:,.2f}`
R:R: `1:{rr_long:.2f}`

🔴 *SHORT* (Confluence: {short_score}/8):
Entry: `${entry:,.2f}`
SL: `${sl_short:,.2f}`
TP1: `${tp1_short:,.2f}`
TP2: `${tp2_short:,.2f}`
R:R: `1:{rr_short:.2f}`

━━━━━━━━━━━━━━━━━━━

📊 *Market Data*:
RSI: `{latest_5m['rsi']:.0f}` | Stoch: `{latest_5m['stoch_k']:.0f}`
MACD: `{latest_5m['macd']:.4f}` | Fear: `{fg}/100`
Funding: `{funding:.4f}%` | L/S: `{ls:.2f}`
Sentiment: `{sentiment:.2f}`

━━━━━━━━━━━━━━━━━━━

📋 *Paste ke Claude Pro untuk detail analisis*
"""
    
    send_telegram(message)
    print(f"✅ [{datetime.now().strftime('%H:%M:%S')}] Signal sent!")
    print(f"💰 {latest_5m['close']:,.2f} | L:{long_score} S:{short_score}")

# ==================== MAIN LOOP ====================

if __name__ == "__main__":
    print("🤖 TELEGRAM TRADING BOT PRO - 5 MENIT")
    print("📱 Update setiap 5 menit ke HP")
    print("🔥 Full Confluence System + Whale + News Sentiment")
    print("\n⚠️  VPN aktif! Tekan Control+C untuk stop\n")
    
    send_telegram("🚀 *Bot Started!*\n5-min signals incoming...")
    
    while True:
        try:
            analyze_and_send("BTCUSDT")
            time.sleep(300)  # 5 MENIT
        except KeyboardInterrupt:
            send_telegram("⏹️ *Bot Stopped*")
            print("\n👋 Stopped.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(30)
