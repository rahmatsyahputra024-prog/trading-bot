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

NTFY_URL = "https://ntfy.sh/Mamat-trading"

CONFIG = {
    "symbol": "BTCUSDT",
    "min_confluence": 6,
    "min_adx": 22,
    "min_rr": 2.0,
    "atr_sl_mult": 1.3,
    "atr_tp_mult": 3.0,
    "min_volume_ratio": 1.2,
}

BULLISH_KW = ['surge','rally','breakout','bullish','adoption','approval','upgrade','ath','etf','accumulation','halving','bullrun']
BEARISH_KW = ['crash','dump','bearish','ban','regulation','hack','scam','sec','sell-off','liquidation','exploit','collapse']
HIGH_IMPACT = ['fed','fomc','cpi','ppi','war','hack']

def get_klines(symbol, interval, limit=500):
    k = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(k, columns=["time","open","high","low","close","volume","close_time","quote_vol","trades","taker_buy_base","taker_buy_quote","ignore"])
    df[["open","high","low","close","volume","taker_buy_base"]] = df[["open","high","low","close","volume","taker_buy_base"]].astype(float)
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
    df["vwap"] = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum() / df["volume"].cumsum()
    df["vwap_dev"] = (df["close"] - df["vwap"]) / df["vwap"] * 100
    df["taker_sell"] = df["volume"] - df["taker_buy_base"]
    df["order_flow"] = (df["taker_buy_base"] - df["taker_sell"]) / df["volume"] * 100
    df["order_flow_ma"] = df["order_flow"].rolling(window=10).mean()
    return df.dropna()

def detect_order_block(df, lookback=20):
    recent = df.tail(lookback)
    high_vol = recent[recent['volume'] > recent['volume_sma'] * 2]
    if len(high_vol) == 0: return {"bull_ob": 0, "bear_ob": 0}
    bull = high_vol[high_vol['close'] > high_vol['open']]
    bear = high_vol[high_vol['close'] < high_vol['open']]
    return {
        "bull_ob": bull['low'].max() if len(bull) > 0 else 0,
        "bear_ob": bear['high'].min() if len(bear) > 0 else 0
    }

def detect_liquidity_sweep(df, lookback=20):
    recent = df.tail(lookback)
    latest = df.iloc[-1]
    prev_high = recent['high'].iloc[:-1].max()
    prev_low = recent['low'].iloc[:-1].min()
    return {
        "swept_high": latest['high'] > prev_high and latest['close'] < prev_high,
        "swept_low": latest['low'] < prev_low and latest['close'] > prev_low
    }

def get_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        d = r.json()["data"][0]
        return {"value": int(d["value"]), "class": d["value_classification"]}
    except: return {"value": 50, "class": "Unknown"}

def get_funding(symbol):
    try:
        r = requests.get(f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}", timeout=10)
        return float(r.json()["lastFundingRate"]) * 100
    except: return 0.0

def get_long_short(symbol):
    try:
        r = requests.get(f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={symbol}&period=15m&limit=1", timeout=10)
        d = r.json()[0]
        return {"long": float(d["longAccount"])*100, "short": float(d["shortAccount"])*100, "ratio": float(d["longShortRatio"])}
    except: return {"long": 50, "short": 50, "ratio": 1.0}

def get_whale(symbol, min_qty=5):
    try:
        r = requests.get(f"https://api.binance.com/api/v3/trades?symbol={symbol}&limit=500", timeout=10)
        buys, sells = 0, 0
        for t in r.json():
            qty = float(t["qty"])
            if qty >= min_qty:
                if t["isBuyerMaker"]: sells += qty
                else: buys += qty
        return {"buy": buys, "sell": sells, "pressure": "BUY" if buys > sells*1.3 else "SELL" if sells > buys*1.3 else "NEUTRAL"}
    except: return {"buy": 0, "sell": 0, "pressure": "NEUTRAL"}

def check_news():
    try:
        feed = feedparser.parse("https://cointelegraph.com/rss")
        high_impact = False
        scores = []
        titles = []
        for entry in feed.entries[:5]:
            title = entry.title[:80]
            tl = title.lower()
            if any(kw in tl for kw in HIGH_IMPACT):
                high_impact = True
            score = analyzer.polarity_scores(tl)['compound']
            bull = sum(1 for kw in BULLISH_KW if kw in tl)
            bear = sum(1 for kw in BEARISH_KW if kw in tl)
            scores.append(score + (bull - bear) * 0.1)
            titles.append(title)
        return {"high_impact": high_impact, "sentiment": sum(scores)/len(scores) if scores else 0, "titles": titles}
    except:
        return {"high_impact": False, "sentiment": 0, "titles": []}

def calculate_confluence(df_5m, df_15m, df_1h, fg, whale, ob, sweep):
    latest_5m = df_5m.iloc[-1]
    latest_15m = df_15m.iloc[-1]
    latest_1h = df_1h.iloc[-1]
    
    long_score, short_score = 0, 0
    signals_long, signals_short = [], []
    
    # 1. MTF Trend
    if latest_1h['ema_50'] > latest_1h['ema_200'] and latest_15m['ema_50'] > latest_15m['ema_200']:
        long_score += 1
        signals_long.append("MTF Trend Aligned BULL")
    elif latest_1h['ema_50'] < latest_1h['ema_200'] and latest_15m['ema_50'] < latest_15m['ema_200']:
        short_score += 1
        signals_short.append("MTF Trend Aligned BEAR")
    
    # 2. EMA Structure
    if latest_5m['ema_9'] > latest_5m['ema_21'] > latest_5m['ema_50']:
        long_score += 1
        signals_long.append("EMA Stack Bullish")
    elif latest_5m['ema_9'] < latest_5m['ema_21'] < latest_5m['ema_50']:
        short_score += 1
        signals_short.append("EMA Stack Bearish")
    
    # 3. RSI
    if latest_5m['rsi'] < 35 and latest_5m['rsi'] > df_5m.iloc[-2]['rsi']:
        long_score += 1
        signals_long.append(f"RSI Oversold+Turning ({latest_5m['rsi']:.0f})")
    elif latest_5m['rsi'] > 65 and latest_5m['rsi'] < df_5m.iloc[-2]['rsi']:
        short_score += 1
        signals_short.append(f"RSI Overbought+Turning ({latest_5m['rsi']:.0f})")
    
    # 4. MACD
    if latest_5m['macd'] > latest_5m['macd_signal']:
        long_score += 1
        signals_long.append("MACD Bullish")
    else:
        short_score += 1
        signals_short.append("MACD Bearish")
    
    # 5. VWAP
    if latest_5m['close'] > latest_5m['vwap']:
        long_score += 1
        signals_long.append(f"Above VWAP ({latest_5m['vwap_dev']:.2f}%)")
    else:
        short_score += 1
        signals_short.append(f"Below VWAP ({latest_5m['vwap_dev']:.2f}%)")
    
    # 6. Order Flow
    if latest_5m['order_flow_ma'] > 10:
        long_score += 1
        signals_long.append(f"Buy Flow Dominant (+{latest_5m['order_flow_ma']:.1f}%)")
    elif latest_5m['order_flow_ma'] < -10:
        short_score += 1
        signals_short.append(f"Sell Flow Dominant ({latest_5m['order_flow_ma']:.1f}%)")
    
    # 7. Whale + Liquidity Sweep
    if whale['pressure'] == "BUY" and sweep['swept_low']:
        long_score += 2
        signals_long.append("WHALE BUY + LIQ SWEPT")
    elif whale['pressure'] == "SELL" and sweep['swept_high']:
        short_score += 2
        signals_short.append("WHALE SELL + LIQ SWEPT")
    elif whale['pressure'] == "BUY":
        long_score += 1
        signals_long.append("Whale Accumulating")
    elif whale['pressure'] == "SELL":
        short_score += 1
        signals_short.append("Whale Distributing")
    
    # 8. Sentiment
    if fg['value'] < 25:
        long_score += 1
        signals_long.append(f"Extreme Fear ({fg['value']})")
    elif fg['value'] > 75:
        short_score += 1
        signals_short.append(f"Extreme Greed ({fg['value']})")
    
    return {"long": long_score, "short": short_score, "sig_long": signals_long, "sig_short": signals_short}

def create_setup(direction, latest, atr, ob):
    entry = latest['close']
    if direction == "LONG":
        sl = min(entry - (atr * CONFIG["atr_sl_mult"]), ob['bull_ob'] * 0.998 if ob['bull_ob'] else entry - atr*2)
        tp1 = entry + (atr * 2.0)
        tp2 = entry + (atr * CONFIG["atr_tp_mult"])
    else:
        sl = max(entry + (atr * CONFIG["atr_sl_mult"]), ob['bear_ob'] * 1.002 if ob['bear_ob'] else entry + atr*2)
        tp1 = entry - (atr * 2.0)
        tp2 = entry - (atr * CONFIG["atr_tp_mult"])
    risk = abs(entry - sl)
    reward = abs(tp2 - entry)
    return {"entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "rr": reward/risk if risk > 0 else 0}

def send_ntfy(title, body, priority="default"):
    try:
        requests.post(
            NTFY_URL,
            data=body.encode('utf-8'),
            headers={"Title": title.encode('utf-8'), "Priority": priority}
        )
        return True
    except Exception as e:
        print(f"Ntfy error: {e}")
        return False

def analyze_and_send():
    symbol = CONFIG["symbol"]
    try:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Scanning {symbol}...")
        
        df_5m = calc_indicators(get_klines(symbol, "5m", 500))
        df_15m = calc_indicators(get_klines(symbol, "15m", 500))
        df_1h = calc_indicators(get_klines(symbol, "1h", 500))
        
        latest = df_5m.iloc[-1]
        atr = latest['atr']
        
        news = check_news()
        fg = get_fear_greed()
        funding = get_funding(symbol)
        ls = get_long_short(symbol)
        whale = get_whale(symbol)
        ob = detect_order_block(df_5m)
        sweep = detect_liquidity_sweep(df_5m)
        
        conf = calculate_confluence(df_5m, df_15m, df_1h, fg, whale, ob, sweep)
        
        support = df_5m["low"].tail(50).min()
        resistance = df_5m["high"].tail(50).max()
        trend_1h = "BULLISH" if df_1h.iloc[-1]['ema_50'] > df_1h.iloc[-1]['ema_200'] else "BEARISH"
        trend_5m = "BULLISH" if latest['ema_9'] > latest['ema_21'] else "BEARISH"
        
        # Determine direction
        if conf['long'] > conf['short']:
            direction = "LONG"
            setup = create_setup("LONG", latest, atr, ob)
            signals = conf['sig_long']
            score = conf['long']
        else:
            direction = "SHORT"
            setup = create_setup("SHORT", latest, atr, ob)
            signals = conf['sig_short']
            score = conf['short']
        
        grade = "A+" if score >= 8 else "A" if score >= 6 else "B" if score >= 4 else "C"
        priority = "high" if score >= 7 else "default"
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        title = f"{direction} {symbol} | ${setup['entry']:,.0f} | {grade} {score}/9"
        
        body = f"""INSTITUTIONAL TRADING SIGNAL
{timestamp} | {symbol} 5M

=== SIGNAL QUALITY ===
Direction: {direction}
Grade: {grade}
Confluence: {score}/9
Priority: {'HIGH' if score >= 7 else 'NORMAL'}

=== TRADE SETUP ===
Entry: ${setup['entry']:,.2f}
Stop Loss: ${setup['sl']:,.2f}
Take Profit 1: ${setup['tp1']:,.2f}
Take Profit 2: ${setup['tp2']:,.2f}
Risk/Reward: 1:{setup['rr']:.2f}

=== PRICE ACTION ===
Price: ${latest['close']:,.2f}
Support: ${support:,.2f}
Resistance: ${resistance:,.2f}
Trend 1H: {trend_1h}
Trend 5M: {trend_5m}

=== MOMENTUM ===
RSI 14: {latest['rsi']:.2f}
RSI 7: {latest['rsi_fast']:.2f}
MACD: {latest['macd']:.6f}
MACD Signal: {latest['macd_signal']:.6f}
Stochastic K: {latest['stoch_k']:.2f}
Stochastic D: {latest['stoch_d']:.2f}
ADX: {latest['adx']:.2f}

=== VOLATILITY ===
BB Upper: ${latest['bb_upper']:,.2f}
BB Lower: ${latest['bb_lower']:,.2f}
ATR: ${atr:.2f}

=== ORDER FLOW ===
VWAP: ${latest['vwap']:,.2f}
VWAP Deviation: {latest['vwap_dev']:.2f}%
Buy/Sell Flow: {latest['order_flow_ma']:.2f}%
Volume: {latest['volume']:,.0f}
Volume MA: {latest['volume_sma']:,.0f}

=== SMART MONEY ===
Bullish Order Block: ${ob['bull_ob']:,.2f}
Bearish Order Block: ${ob['bear_ob']:,.2f}
Liquidity Swept High: {'YES' if sweep['swept_high'] else 'NO'}
Liquidity Swept Low: {'YES' if sweep['swept_low'] else 'NO'}
Whale Pressure: {whale['pressure']}
Whale Buy Volume: {whale['buy']:.2f} BTC
Whale Sell Volume: {whale['sell']:.2f} BTC

=== SENTIMENT ===
Fear & Greed: {fg['value']}/100 ({fg['class']})
Funding Rate: {funding:.4f}%
Long/Short Ratio: {ls['ratio']:.2f}
Long: {ls['long']:.1f}% | Short: {ls['short']:.1f}%
News Sentiment: {news['sentiment']:.2f}
High Impact News: {'YES' if news['high_impact'] else 'NO'}

=== CONFLUENCE SIGNALS ({len(signals)}) ===
"""
        for i, s in enumerate(signals, 1):
            body += f"{i}. {s}\n"
        
        if news['titles']:
            body += f"\n=== LATEST NEWS ===\n"
            for i, t in enumerate(news['titles'][:3], 1):
                body += f"{i}. {t}\n"
        
        body += f"""
=========================================
PROMPT UNTUK CLAUDE PRO:
=========================================

Sebagai expert trader profesional institusional, tolong analisis data trading BTCUSDT 5M di atas dan berikan:

1. VALIDASI SINYAL
   - Apakah setup {direction} ini valid?
   - Berapa tingkat kepercayaan (0-100%)?
   - Apakah ada red flag yang tidak terlihat dari data?

2. ANALISIS MULTI-FAKTOR
   - Konfirmasi trend multi-timeframe (5M, 15M, 1H)
   - Konsistensi smart money dan whale behavior
   - Apakah order flow mendukung direction?
   - Interpretasi sentimen dan news impact

3. REFINEMENT ENTRY
   - Apakah entry ${setup['entry']:,.2f} optimal?
   - Jika tidak, rekomendasi entry price yang lebih baik
   - Timing entry (sekarang/tunggu pullback/tunggu breakout)

4. RISK MANAGEMENT
   - Validasi Stop Loss di ${setup['sl']:,.2f}
   - Apakah Take Profit ${setup['tp2']:,.2f} realistis?
   - Saran position sizing (berapa % dari modal)
   - Risk/Reward optimal

5. SKENARIO
   - Skenario BULLISH: apa target dan trigger?
   - Skenario BEARISH: level invalidasi
   - Skenario SIDEWAYS: apa yang harus dilakukan?

6. KEPUTUSAN FINAL
   - EKSEKUSI SEKARANG / TUNGGU / SKIP
   - Alasan 1-2 kalimat
   - Confidence level
   - Time frame trade (scalp 15m / 1-2 jam / swing)

Berikan jawaban dalam format terstruktur dan mudah dibaca.
"""
        
        send_ntfy(title, body, priority)
        print(f"OK Signal sent! {direction} @ ${setup['entry']:,.0f} | Grade {grade} ({score}/9)")
        print(f"   R:R 1:{setup['rr']:.2f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("="*60)
    print("INSTITUTIONAL TRADING BOT v8.0")
    print("="*60)
    print("Strategy: Smart Money + MTF Confluence")
    print("Signal delivery: Ntfy HP")
    print("Timeframe: 5M scalping")
    print("Auto Prompt: Claude Pro analysis included")
    print("\nVPN must be active!")
    print("Press Control+C to stop\n")
    
    send_ntfy("BOT STARTED", "Institutional Trading Bot v8.0 is LIVE\n5M scalping with smart money analysis\nSignals coming every 5 minutes")
    
    while True:
        try:
            analyze_and_send()
            print("\nWait 5 minutes...\n")
            time.sleep(300)
        except KeyboardInterrupt:
            send_ntfy("BOT STOPPED", "Bot telah dimatikan")
            print("\nStopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(30)
