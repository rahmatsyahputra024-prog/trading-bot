import os
import time
import requests
import feedparser
import numpy as np
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
    "symbols": ["BTCUSDT", "SOLUSDT"],
    "min_confluence": 10,
    "min_adx": 20,
    "atr_sl_mult": 1.3,
    "atr_tp_mult": 3.0,
}

BULLISH_KW = ['surge','rally','breakout','bullish','adoption','approval','upgrade','ath','etf','accumulation','halving','bullrun']
BEARISH_KW = ['crash','dump','bearish','ban','regulation','hack','scam','sec','sell-off','liquidation','exploit','collapse']
HIGH_IMPACT = ['fed','fomc','cpi','ppi','war','hack']

def get_klines(symbol, interval, limit=500):
    k = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(k, columns=["time","open","high","low","close","volume","close_time","quote_vol","trades","taker_buy_base","taker_buy_quote","ignore"])
    df[["open","high","low","close","volume","taker_buy_base"]] = df[["open","high","low","close","volume","taker_buy_base"]].astype(float)
    return df

def calc_all_indicators(df):
    df["ema_9"] = ta.trend.EMAIndicator(df["close"], 9).ema_indicator()
    df["ema_21"] = ta.trend.EMAIndicator(df["close"], 21).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(df["close"], 50).ema_indicator()
    df["ema_100"] = ta.trend.EMAIndicator(df["close"], 100).ema_indicator()
    df["ema_200"] = ta.trend.EMAIndicator(df["close"], 200).ema_indicator()
    df["sma_50"] = ta.trend.SMAIndicator(df["close"], 50).sma_indicator()
    df["sma_200"] = ta.trend.SMAIndicator(df["close"], 200).sma_indicator()
    
    ichi = ta.trend.IchimokuIndicator(df["high"], df["low"])
    df["ichi_a"] = ichi.ichimoku_a()
    df["ichi_b"] = ichi.ichimoku_b()
    df["ichi_base"] = ichi.ichimoku_base_line()
    df["ichi_conv"] = ichi.ichimoku_conversion_line()
    
    df["psar"] = ta.trend.PSARIndicator(df["high"], df["low"], df["close"]).psar()
    
    donch = ta.volatility.DonchianChannel(df["high"], df["low"], df["close"], 20)
    df["donch_upper"] = donch.donchian_channel_hband()
    df["donch_lower"] = donch.donchian_channel_lband()
    df["donch_mid"] = donch.donchian_channel_mband()
    
    hl2 = (df["high"] + df["low"]) / 2
    atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], 10).average_true_range()
    df["st_upper"] = hl2 + (3.0 * atr)
    df["st_lower"] = hl2 - (3.0 * atr)
    
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
    df["rsi_fast"] = ta.momentum.RSIIndicator(df["close"], 7).rsi()
    
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    
    stoch_rsi = ta.momentum.StochRSIIndicator(df["close"], 14)
    df["stoch_k"] = stoch_rsi.stochrsi_k() * 100
    df["stoch_d"] = stoch_rsi.stochrsi_d() * 100
    
    df["williams_r"] = ta.momentum.WilliamsRIndicator(df["high"], df["low"], df["close"], 14).williams_r()
    df["cci"] = ta.trend.CCIIndicator(df["high"], df["low"], df["close"], 20).cci()
    df["roc"] = ta.momentum.ROCIndicator(df["close"], 12).roc()
    df["awesome"] = ta.momentum.AwesomeOscillatorIndicator(df["high"], df["low"]).awesome_oscillator()
    
    bb = ta.volatility.BollingerBands(df["close"], 20, 2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"] * 100
    
    kc = ta.volatility.KeltnerChannel(df["high"], df["low"], df["close"], 20)
    df["kc_upper"] = kc.keltner_channel_hband()
    df["kc_lower"] = kc.keltner_channel_lband()
    df["kc_mid"] = kc.keltner_channel_mband()
    
    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], 14).average_true_range()
    
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
    df["obv_ma"] = df["obv"].rolling(20).mean()
    df["cmf"] = ta.volume.ChaikinMoneyFlowIndicator(df["high"], df["low"], df["close"], df["volume"], 20).chaikin_money_flow()
    df["mfi"] = ta.volume.MFIIndicator(df["high"], df["low"], df["close"], df["volume"], 14).money_flow_index()
    df["ad"] = ta.volume.AccDistIndexIndicator(df["high"], df["low"], df["close"], df["volume"]).acc_dist_index()
    df["volume_sma"] = df["volume"].rolling(20).mean()
    
    df["vwap"] = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum() / df["volume"].cumsum()
    df["vwap_dev"] = (df["close"] - df["vwap"]) / df["vwap"] * 100
    
    df["taker_sell"] = df["volume"] - df["taker_buy_base"]
    df["order_flow"] = (df["taker_buy_base"] - df["taker_sell"]) / df["volume"] * 100
    df["order_flow_ma"] = df["order_flow"].rolling(10).mean()
    
    df["adx"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], 14).adx()
    df["di_plus"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], 14).adx_pos()
    df["di_minus"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], 14).adx_neg()
    
    return df.dropna()

def detect_smart_money(df, lookback=20):
    recent = df.tail(lookback)
    latest = df.iloc[-1]
    
    high_vol = recent[recent['volume'] > recent['volume_sma'] * 2]
    bull_ob = high_vol[high_vol['close'] > high_vol['open']]['low'].max() if len(high_vol) > 0 else 0
    bear_ob = high_vol[high_vol['close'] < high_vol['open']]['high'].min() if len(high_vol) > 0 else 0
    
    prev_high = recent['high'].iloc[:-1].max()
    prev_low = recent['low'].iloc[:-1].min()
    swept_high = latest['high'] > prev_high and latest['close'] < prev_high
    swept_low = latest['low'] < prev_low and latest['close'] > prev_low
    
    fvg_bull = False
    fvg_bear = False
    if len(df) >= 3:
        c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        if c1['high'] < c3['low']:
            fvg_bull = True
        if c1['low'] > c3['high']:
            fvg_bear = True
    
    highs = recent['high'].values
    lows = recent['low'].values
    
    higher_highs = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
    lower_lows = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])
    
    if higher_highs > lower_lows + 3:
        structure = "UPTREND (HH+HL)"
    elif lower_lows > higher_highs + 3:
        structure = "DOWNTREND (LL+LH)"
    else:
        structure = "RANGING"
    
    return {
        "bull_ob": bull_ob,
        "bear_ob": bear_ob,
        "swept_high": swept_high,
        "swept_low": swept_low,
        "fvg_bull": fvg_bull,
        "fvg_bear": fvg_bear,
        "structure": structure
    }

def calc_fibonacci(df, lookback=50):
    recent = df.tail(lookback)
    high = recent['high'].max()
    low = recent['low'].min()
    diff = high - low
    return {
        "fib_0": high,
        "fib_236": high - (diff * 0.236),
        "fib_382": high - (diff * 0.382),
        "fib_500": high - (diff * 0.500),
        "fib_618": high - (diff * 0.618),
        "fib_786": high - (diff * 0.786),
        "fib_1": low
    }

def calc_pivot_points(df):
    prev = df.iloc[-2]
    pp = (prev['high'] + prev['low'] + prev['close']) / 3
    r1 = 2 * pp - prev['low']
    s1 = 2 * pp - prev['high']
    r2 = pp + (prev['high'] - prev['low'])
    s2 = pp - (prev['high'] - prev['low'])
    return {"pp": pp, "r1": r1, "r2": r2, "s1": s1, "s2": s2}

def detect_candle_pattern(df):
    c1, c2 = df.iloc[-2], df.iloc[-1]
    patterns = []
    
    if c2['close'] > c2['open'] and c1['close'] < c1['open']:
        if c2['close'] > c1['open'] and c2['open'] < c1['close']:
            patterns.append("Bullish Engulfing")
    
    if c2['close'] < c2['open'] and c1['close'] > c1['open']:
        if c2['close'] < c1['open'] and c2['open'] > c1['close']:
            patterns.append("Bearish Engulfing")
    
    body = abs(c2['close'] - c2['open'])
    wick = c2['high'] - c2['low']
    if wick > 0 and body / wick < 0.1:
        patterns.append("Doji")
    
    lower_wick = min(c2['open'], c2['close']) - c2['low']
    upper_wick = c2['high'] - max(c2['open'], c2['close'])
    if lower_wick > body * 2 and upper_wick < body:
        patterns.append("Hammer")
    
    if upper_wick > body * 2 and lower_wick < body:
        patterns.append("Shooting Star")
    
    return patterns if patterns else ["None"]

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

def get_open_interest(symbol):
    try:
        r = requests.get(f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}", timeout=10)
        return float(r.json()["openInterest"])
    except: return 0.0

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
        scores, titles = [], []
        high_impact = False
        for entry in feed.entries[:5]:
            title = entry.title[:80]
            tl = title.lower()
            if any(kw in tl for kw in HIGH_IMPACT): high_impact = True
            score = analyzer.polarity_scores(tl)['compound']
            bull = sum(1 for kw in BULLISH_KW if kw in tl)
            bear = sum(1 for kw in BEARISH_KW if kw in tl)
            scores.append(score + (bull - bear) * 0.1)
            titles.append(title)
        return {"high_impact": high_impact, "sentiment": sum(scores)/len(scores) if scores else 0, "titles": titles}
    except:
        return {"high_impact": False, "sentiment": 0, "titles": []}

def calculate_confluence_ultimate(df_5m, df_15m, df_1h, fg, whale, sm):
    latest = df_5m.iloc[-1]
    latest_15m = df_15m.iloc[-1]
    latest_1h = df_1h.iloc[-1]
    
    L, S = 0, 0
    sig_l, sig_s = [], []
    
    if latest_1h['ema_50'] > latest_1h['ema_200'] and latest_15m['ema_50'] > latest_15m['ema_200']:
        L += 2; sig_l.append("MTF Bull Aligned")
    elif latest_1h['ema_50'] < latest_1h['ema_200'] and latest_15m['ema_50'] < latest_15m['ema_200']:
        S += 2; sig_s.append("MTF Bear Aligned")
    
    if latest['ema_9'] > latest['ema_21'] > latest['ema_50']:
        L += 1; sig_l.append("EMA Stack Bull")
    elif latest['ema_9'] < latest['ema_21'] < latest['ema_50']:
        S += 1; sig_s.append("EMA Stack Bear")
    
    if latest['close'] > latest['ichi_a'] and latest['close'] > latest['ichi_b']:
        L += 1; sig_l.append("Above Ichi Cloud")
    elif latest['close'] < latest['ichi_a'] and latest['close'] < latest['ichi_b']:
        S += 1; sig_s.append("Below Ichi Cloud")
    
    if latest['close'] > latest['psar']:
        L += 1; sig_l.append("PSAR Bull")
    else:
        S += 1; sig_s.append("PSAR Bear")
    
    if 30 < latest['rsi'] < 50 and latest['rsi'] > df_5m.iloc[-2]['rsi']:
        L += 1; sig_l.append(f"RSI Recovery ({latest['rsi']:.0f})")
    elif 50 < latest['rsi'] < 70 and latest['rsi'] < df_5m.iloc[-2]['rsi']:
        S += 1; sig_s.append(f"RSI Declining ({latest['rsi']:.0f})")
    
    if latest['macd'] > latest['macd_signal'] and latest['macd_hist'] > 0:
        L += 1; sig_l.append("MACD Bull+Hist")
    elif latest['macd'] < latest['macd_signal'] and latest['macd_hist'] < 0:
        S += 1; sig_s.append("MACD Bear+Hist")
    
    if latest['stoch_k'] < 20 and latest['stoch_k'] > latest['stoch_d']:
        L += 1; sig_l.append("Stoch Oversold+Cross")
    elif latest['stoch_k'] > 80 and latest['stoch_k'] < latest['stoch_d']:
        S += 1; sig_s.append("Stoch Overbought+Cross")
    
    if latest['williams_r'] < -80:
        L += 1; sig_l.append("Williams Oversold")
    elif latest['williams_r'] > -20:
        S += 1; sig_s.append("Williams Overbought")
    
    if latest['cci'] < -100:
        L += 1; sig_l.append(f"CCI Oversold ({latest['cci']:.0f})")
    elif latest['cci'] > 100:
        S += 1; sig_s.append(f"CCI Overbought ({latest['cci']:.0f})")
    
    if latest['mfi'] < 20:
        L += 1; sig_l.append("MFI Oversold")
    elif latest['mfi'] > 80:
        S += 1; sig_s.append("MFI Overbought")
    
    if latest['close'] < latest['bb_lower']:
        L += 1; sig_l.append("Below BB Lower")
    elif latest['close'] > latest['bb_upper']:
        S += 1; sig_s.append("Above BB Upper")
    
    if latest['close'] > latest['vwap'] and -0.5 < latest['vwap_dev'] < 1.5:
        L += 1; sig_l.append(f"Above VWAP (+{latest['vwap_dev']:.2f}%)")
    elif latest['close'] < latest['vwap'] and -1.5 < latest['vwap_dev'] < 0.5:
        S += 1; sig_s.append(f"Below VWAP ({latest['vwap_dev']:.2f}%)")
    
    if latest['volume'] > latest['volume_sma'] * 1.5:
        if latest['close'] > latest['open']:
            L += 1; sig_l.append("High Vol + Green")
        else:
            S += 1; sig_s.append("High Vol + Red")
    
    if latest['cmf'] > 0.1:
        L += 1; sig_l.append(f"CMF Positive ({latest['cmf']:.2f})")
    elif latest['cmf'] < -0.1:
        S += 1; sig_s.append(f"CMF Negative ({latest['cmf']:.2f})")
    
    if latest['obv'] > latest['obv_ma']:
        L += 1; sig_l.append("OBV Rising")
    else:
        S += 1; sig_s.append("OBV Falling")
    
    if latest['order_flow_ma'] > 15:
        L += 1; sig_l.append(f"Buy Flow +{latest['order_flow_ma']:.1f}%")
    elif latest['order_flow_ma'] < -15:
        S += 1; sig_s.append(f"Sell Flow {latest['order_flow_ma']:.1f}%")
    
    if latest['adx'] > 25:
        if latest['di_plus'] > latest['di_minus']:
            L += 1; sig_l.append(f"ADX Bull ({latest['adx']:.0f})")
        else:
            S += 1; sig_s.append(f"ADX Bear ({latest['adx']:.0f})")
    
    if whale['pressure'] == "BUY" and (sm['swept_low'] or sm['fvg_bull']):
        L += 2; sig_l.append("WHALE BUY + SMC")
    elif whale['pressure'] == "SELL" and (sm['swept_high'] or sm['fvg_bear']):
        S += 2; sig_s.append("WHALE SELL + SMC")
    elif whale['pressure'] == "BUY":
        L += 1; sig_l.append("Whale Buying")
    elif whale['pressure'] == "SELL":
        S += 1; sig_s.append("Whale Selling")
    
    if "UPTREND" in sm['structure']:
        L += 1; sig_l.append("Structure: HH+HL")
    elif "DOWNTREND" in sm['structure']:
        S += 1; sig_s.append("Structure: LL+LH")
    
    if fg['value'] < 25:
        L += 1; sig_l.append(f"Extreme Fear ({fg['value']})")
    elif fg['value'] > 75:
        S += 1; sig_s.append(f"Extreme Greed ({fg['value']})")
    
    return {"long": L, "short": S, "sig_long": sig_l, "sig_short": sig_s}

def create_setup(direction, latest, atr, sm, fib, pivot):
    entry = latest['close']
    if direction == "LONG":
        sl_candidates = [entry - (atr * CONFIG["atr_sl_mult"])]
        if sm['bull_ob']: sl_candidates.append(sm['bull_ob'] * 0.998)
        if pivot['s1'] < entry: sl_candidates.append(pivot['s1'])
        sl = min(sl_candidates)
        tp1 = entry + (atr * 2.0)
        tp2 = entry + (atr * CONFIG["atr_tp_mult"])
        tp3 = min(pivot['r2'], entry + (atr * 5.0))
    else:
        sl_candidates = [entry + (atr * CONFIG["atr_sl_mult"])]
        if sm['bear_ob']: sl_candidates.append(sm['bear_ob'] * 1.002)
        if pivot['r1'] > entry: sl_candidates.append(pivot['r1'])
        sl = max(sl_candidates)
        tp1 = entry - (atr * 2.0)
        tp2 = entry - (atr * CONFIG["atr_tp_mult"])
        tp3 = max(pivot['s2'], entry - (atr * 5.0))
    risk = abs(entry - sl)
    reward = abs(tp2 - entry)
    return {"entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3, "rr": reward/risk if risk > 0 else 0}

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

def analyze_and_send(symbol):
    try:
        print(f">>> Analyzing {symbol}...")
        
        df_5m = calc_all_indicators(get_klines(symbol, "5m", 500))
        df_15m = calc_all_indicators(get_klines(symbol, "15m", 500))
        df_1h = calc_all_indicators(get_klines(symbol, "1h", 500))
        
        latest = df_5m.iloc[-1]
        atr = latest['atr']
        
        fg = get_fear_greed()
        funding = get_funding(symbol)
        ls = get_long_short(symbol)
        oi = get_open_interest(symbol)
        whale = get_whale(symbol)
        news = check_news()
        sm = detect_smart_money(df_5m)
        fib = calc_fibonacci(df_5m)
        pivot = calc_pivot_points(df_1h)
        patterns = detect_candle_pattern(df_5m)
        
        conf = calculate_confluence_ultimate(df_5m, df_15m, df_1h, fg, whale, sm)
        
        support = df_5m["low"].tail(50).min()
        resistance = df_5m["high"].tail(50).max()
        trend_1h = "BULLISH" if df_1h.iloc[-1]['ema_50'] > df_1h.iloc[-1]['ema_200'] else "BEARISH"
        trend_5m = "BULLISH" if latest['ema_9'] > latest['ema_21'] else "BEARISH"
        
        if conf['long'] > conf['short']:
            direction = "LONG"
            setup = create_setup("LONG", latest, atr, sm, fib, pivot)
            signals = conf['sig_long']
            score = conf['long']
        else:
            direction = "SHORT"
            setup = create_setup("SHORT", latest, atr, sm, fib, pivot)
            signals = conf['sig_short']
            score = conf['short']
        
        grade = "A+" if score >= 14 else "A" if score >= 10 else "B" if score >= 7 else "C"
        priority = "high" if score >= 12 else "default"
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        title = f"{direction} {symbol} | ${setup['entry']:,.0f} | {grade} {score}/20"
        
        body = f"""ULTIMATE TRADING SIGNAL v9.0
{timestamp} | {symbol} 5M

=== SIGNAL QUALITY ===
Direction: {direction}
Grade: {grade}
Confluence: {score}/20
Priority: {'HIGH' if score >= 12 else 'NORMAL'}

=== TRADE SETUP ===
Entry: ${setup['entry']:,.2f}
Stop Loss: ${setup['sl']:,.2f}
TP1 (50%): ${setup['tp1']:,.2f}
TP2 (30%): ${setup['tp2']:,.2f}
TP3 (20%): ${setup['tp3']:,.2f}
Risk/Reward: 1:{setup['rr']:.2f}

=== PRICE ACTION ===
Price: ${latest['close']:,.2f}
Support: ${support:,.2f}
Resistance: ${resistance:,.2f}
Trend 1H: {trend_1h}
Trend 5M: {trend_5m}
Market Structure: {sm['structure']}
Candle Pattern: {', '.join(patterns)}

=== TREND INDICATORS ===
EMA 9: ${latest['ema_9']:,.2f}
EMA 21: ${latest['ema_21']:,.2f}
EMA 50: ${latest['ema_50']:,.2f}
EMA 100: ${latest['ema_100']:,.2f}
EMA 200: ${latest['ema_200']:,.2f}
SMA 50: ${latest['sma_50']:,.2f}
SMA 200: ${latest['sma_200']:,.2f}
Ichimoku A: ${latest['ichi_a']:,.2f}
Ichimoku B: ${latest['ichi_b']:,.2f}
PSAR: ${latest['psar']:,.2f}
Donchian U/L: ${latest['donch_upper']:,.2f} / ${latest['donch_lower']:,.2f}

=== MOMENTUM ===
RSI 14: {latest['rsi']:.2f}
RSI 7: {latest['rsi_fast']:.2f}
MACD: {latest['macd']:.6f}
MACD Signal: {latest['macd_signal']:.6f}
MACD Hist: {latest['macd_hist']:.6f}
Stoch K/D: {latest['stoch_k']:.2f} / {latest['stoch_d']:.2f}
Williams %R: {latest['williams_r']:.2f}
CCI: {latest['cci']:.2f}
ROC: {latest['roc']:.2f}%
Awesome: {latest['awesome']:.2f}

=== VOLATILITY ===
BB Upper: ${latest['bb_upper']:,.2f}
BB Lower: ${latest['bb_lower']:,.2f}
BB Width: {latest['bb_width']:.2f}%
Keltner U/L: ${latest['kc_upper']:,.2f} / ${latest['kc_lower']:,.2f}
ATR: ${atr:.2f}

=== VOLUME ANALYSIS ===
Volume: {latest['volume']:,.0f}
Volume MA: {latest['volume_sma']:,.0f}
OBV: {latest['obv']:,.0f}
CMF: {latest['cmf']:.3f}
MFI: {latest['mfi']:.2f}
Order Flow: {latest['order_flow_ma']:.2f}%

=== VWAP ===
VWAP: ${latest['vwap']:,.2f}
VWAP Deviation: {latest['vwap_dev']:.2f}%

=== TREND STRENGTH ===
ADX: {latest['adx']:.2f}
DI+: {latest['di_plus']:.2f}
DI-: {latest['di_minus']:.2f}

=== SMART MONEY ===
Market Structure: {sm['structure']}
Bullish OB: ${sm['bull_ob']:,.2f}
Bearish OB: ${sm['bear_ob']:,.2f}
Swept High: {'YES' if sm['swept_high'] else 'NO'}
Swept Low: {'YES' if sm['swept_low'] else 'NO'}
FVG Bullish: {'YES' if sm['fvg_bull'] else 'NO'}
FVG Bearish: {'YES' if sm['fvg_bear'] else 'NO'}

=== WHALE ACTIVITY ===
Pressure: {whale['pressure']}
Buy Volume: {whale['buy']:.2f}
Sell Volume: {whale['sell']:.2f}

=== FIBONACCI LEVELS ===
Fib 0.236: ${fib['fib_236']:,.2f}
Fib 0.382: ${fib['fib_382']:,.2f}
Fib 0.500: ${fib['fib_500']:,.2f}
Fib 0.618: ${fib['fib_618']:,.2f}
Fib 0.786: ${fib['fib_786']:,.2f}

=== PIVOT POINTS (1H) ===
R2: ${pivot['r2']:,.2f}
R1: ${pivot['r1']:,.2f}
PP: ${pivot['pp']:,.2f}
S1: ${pivot['s1']:,.2f}
S2: ${pivot['s2']:,.2f}

=== FUTURES DATA ===
Funding Rate: {funding:.4f}%
Open Interest: {oi:,.0f}
Long/Short Ratio: {ls['ratio']:.2f}
Long: {ls['long']:.1f}% | Short: {ls['short']:.1f}%

=== SENTIMENT ===
Fear & Greed: {fg['value']}/100 ({fg['class']})
News Sentiment: {news['sentiment']:.2f}
High Impact News: {'YES' if news['high_impact'] else 'NO'}

=== ACTIVE SIGNALS ({len(signals)}) ===
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

Sebagai expert trader profesional institusional dengan pengalaman 15 tahun di hedge fund, tolong analisis data trading {symbol} 5M scalping di atas dengan komprehensif:

1. VALIDASI SETUP {direction}
   - Apakah setup ini valid dengan {score}/20 confluence?
   - Tingkat kepercayaan setup (0-100%)?
   - Identifikasi red flags atau konflik antar indikator

2. ANALISIS TEKNIKAL MULTI-INDIKATOR
   - Konfirmasi trend (EMA/SMA/Ichimoku/PSAR)
   - Momentum (RSI, MACD, Stoch, Williams, CCI, MFI)
   - Volatility (BB, Keltner, ATR) - apakah trending atau ranging?
   - Volume (OBV, CMF, MFI) - apakah mendukung direction?
   - VWAP & Order Flow - institutional bias?
   - ADX & DI - kekuatan trend?

3. SMART MONEY ANALYSIS
   - Order Block behavior
   - Liquidity Sweep interpretation
   - Fair Value Gap implication
   - Market Structure validity
   - Whale activity correlation

4. KEY LEVELS
   - Fibonacci level yang paling relevan sebagai target/support
   - Pivot Points untuk scalping
   - Support/Resistance paling kuat

5. SENTIMENT & MACRO
   - Fear & Greed impact
   - Funding rate signal (kontrarian/trending)
   - Long/Short ratio interpretation
   - News impact ke harga

6. REFINEMENT ENTRY
   - Apakah entry ${setup['entry']:,.2f} optimal?
   - Rekomendasi entry lebih baik
   - Timing: sekarang/pullback/breakout

7. RISK MANAGEMENT
   - Validasi SL di ${setup['sl']:,.2f}
   - TP realistis: ${setup['tp1']:,.2f} / ${setup['tp2']:,.2f} / ${setup['tp3']:,.2f}
   - Position sizing saran (% modal)
   - Risk per trade max

8. SKENARIO
   - Skenario Bullish: target & trigger
   - Skenario Bearish: invalidasi & exit
   - Skenario Sideways: action plan

9. KEPUTUSAN FINAL
   - EKSEKUSI SEKARANG / TUNGGU KONFIRMASI / SKIP
   - Confidence level 0-100%
   - Time frame: scalp 15m / 1-2 jam / swing
   - Alasan 3 poin utama

Berikan analisis dalam format terstruktur dan actionable.
"""
        
        send_ntfy(title, body, priority)
        print(f"   OK {direction} @ ${setup['entry']:,.0f} | {grade} ({score}/20)")
        
    except Exception as e:
        print(f"   Error: {e}")

if __name__ == "__main__":
    print("="*60)
    print("ULTIMATE TRADING BOT v9.0 - MULTI COIN")
    print("="*60)
    print(f"Coins: {', '.join(CONFIG['symbols'])}")
    print("Signal delivery: Ntfy HP")
    print("Timeframe: 5M scalping")
    print("\nVPN must be active!")
    print("Press Control+C to stop\n")
    
    send_ntfy("ULTIMATE BOT STARTED", f"Ultimate Trading Bot v9.0 is LIVE\nCoins: {', '.join(CONFIG['symbols'])}\n20 indicators aktif")
    
    while True:
        try:
            for symbol in CONFIG["symbols"]:
                analyze_and_send(symbol)
                time.sleep(3)
            print("\nWait 5 minutes for next scan...\n")
            time.sleep(300)
        except KeyboardInterrupt:
            send_ntfy("BOT STOPPED", "Bot telah dimatikan")
            print("\nStopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(30)
ENDOFFILEcat > ultimate_bot.py << 'ENDOFFILE'
import os
import time
import requests
import feedparser
import numpy as np
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
    "symbols": ["BTCUSDT", "SOLUSDT"],
    "min_confluence": 10,
    "min_adx": 20,
    "atr_sl_mult": 1.3,
    "atr_tp_mult": 3.0,
}

BULLISH_KW = ['surge','rally','breakout','bullish','adoption','approval','upgrade','ath','etf','accumulation','halving','bullrun']
BEARISH_KW = ['crash','dump','bearish','ban','regulation','hack','scam','sec','sell-off','liquidation','exploit','collapse']
HIGH_IMPACT = ['fed','fomc','cpi','ppi','war','hack']

def get_klines(symbol, interval, limit=500):
    k = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(k, columns=["time","open","high","low","close","volume","close_time","quote_vol","trades","taker_buy_base","taker_buy_quote","ignore"])
    df[["open","high","low","close","volume","taker_buy_base"]] = df[["open","high","low","close","volume","taker_buy_base"]].astype(float)
    return df

def calc_all_indicators(df):
    df["ema_9"] = ta.trend.EMAIndicator(df["close"], 9).ema_indicator()
    df["ema_21"] = ta.trend.EMAIndicator(df["close"], 21).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(df["close"], 50).ema_indicator()
    df["ema_100"] = ta.trend.EMAIndicator(df["close"], 100).ema_indicator()
    df["ema_200"] = ta.trend.EMAIndicator(df["close"], 200).ema_indicator()
    df["sma_50"] = ta.trend.SMAIndicator(df["close"], 50).sma_indicator()
    df["sma_200"] = ta.trend.SMAIndicator(df["close"], 200).sma_indicator()
    
    ichi = ta.trend.IchimokuIndicator(df["high"], df["low"])
    df["ichi_a"] = ichi.ichimoku_a()
    df["ichi_b"] = ichi.ichimoku_b()
    df["ichi_base"] = ichi.ichimoku_base_line()
    df["ichi_conv"] = ichi.ichimoku_conversion_line()
    
    df["psar"] = ta.trend.PSARIndicator(df["high"], df["low"], df["close"]).psar()
    
    donch = ta.volatility.DonchianChannel(df["high"], df["low"], df["close"], 20)
    df["donch_upper"] = donch.donchian_channel_hband()
    df["donch_lower"] = donch.donchian_channel_lband()
    df["donch_mid"] = donch.donchian_channel_mband()
    
    hl2 = (df["high"] + df["low"]) / 2
    atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], 10).average_true_range()
    df["st_upper"] = hl2 + (3.0 * atr)
    df["st_lower"] = hl2 - (3.0 * atr)
    
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
    df["rsi_fast"] = ta.momentum.RSIIndicator(df["close"], 7).rsi()
    
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    
    stoch_rsi = ta.momentum.StochRSIIndicator(df["close"], 14)
    df["stoch_k"] = stoch_rsi.stochrsi_k() * 100
    df["stoch_d"] = stoch_rsi.stochrsi_d() * 100
    
    df["williams_r"] = ta.momentum.WilliamsRIndicator(df["high"], df["low"], df["close"], 14).williams_r()
    df["cci"] = ta.trend.CCIIndicator(df["high"], df["low"], df["close"], 20).cci()
    df["roc"] = ta.momentum.ROCIndicator(df["close"], 12).roc()
    df["awesome"] = ta.momentum.AwesomeOscillatorIndicator(df["high"], df["low"]).awesome_oscillator()
    
    bb = ta.volatility.BollingerBands(df["close"], 20, 2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"] * 100
    
    kc = ta.volatility.KeltnerChannel(df["high"], df["low"], df["close"], 20)
    df["kc_upper"] = kc.keltner_channel_hband()
    df["kc_lower"] = kc.keltner_channel_lband()
    df["kc_mid"] = kc.keltner_channel_mband()
    
    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], 14).average_true_range()
    
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
    df["obv_ma"] = df["obv"].rolling(20).mean()
    df["cmf"] = ta.volume.ChaikinMoneyFlowIndicator(df["high"], df["low"], df["close"], df["volume"], 20).chaikin_money_flow()
    df["mfi"] = ta.volume.MFIIndicator(df["high"], df["low"], df["close"], df["volume"], 14).money_flow_index()
    df["ad"] = ta.volume.AccDistIndexIndicator(df["high"], df["low"], df["close"], df["volume"]).acc_dist_index()
    df["volume_sma"] = df["volume"].rolling(20).mean()
    
    df["vwap"] = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum() / df["volume"].cumsum()
    df["vwap_dev"] = (df["close"] - df["vwap"]) / df["vwap"] * 100
    
    df["taker_sell"] = df["volume"] - df["taker_buy_base"]
    df["order_flow"] = (df["taker_buy_base"] - df["taker_sell"]) / df["volume"] * 100
    df["order_flow_ma"] = df["order_flow"].rolling(10).mean()
    
    df["adx"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], 14).adx()
    df["di_plus"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], 14).adx_pos()
    df["di_minus"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], 14).adx_neg()
    
    return df.dropna()

def detect_smart_money(df, lookback=20):
    recent = df.tail(lookback)
    latest = df.iloc[-1]
    
    high_vol = recent[recent['volume'] > recent['volume_sma'] * 2]
    bull_ob = high_vol[high_vol['close'] > high_vol['open']]['low'].max() if len(high_vol) > 0 else 0
    bear_ob = high_vol[high_vol['close'] < high_vol['open']]['high'].min() if len(high_vol) > 0 else 0
    
    prev_high = recent['high'].iloc[:-1].max()
    prev_low = recent['low'].iloc[:-1].min()
    swept_high = latest['high'] > prev_high and latest['close'] < prev_high
    swept_low = latest['low'] < prev_low and latest['close'] > prev_low
    
    fvg_bull = False
    fvg_bear = False
    if len(df) >= 3:
        c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        if c1['high'] < c3['low']:
            fvg_bull = True
        if c1['low'] > c3['high']:
            fvg_bear = True
    
    highs = recent['high'].values
    lows = recent['low'].values
    
    higher_highs = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
    lower_lows = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])
    
    if higher_highs > lower_lows + 3:
        structure = "UPTREND (HH+HL)"
    elif lower_lows > higher_highs + 3:
        structure = "DOWNTREND (LL+LH)"
    else:
        structure = "RANGING"
    
    return {
        "bull_ob": bull_ob,
        "bear_ob": bear_ob,
        "swept_high": swept_high,
        "swept_low": swept_low,
        "fvg_bull": fvg_bull,
        "fvg_bear": fvg_bear,
        "structure": structure
    }

def calc_fibonacci(df, lookback=50):
    recent = df.tail(lookback)
    high = recent['high'].max()
    low = recent['low'].min()
    diff = high - low
    return {
        "fib_0": high,
        "fib_236": high - (diff * 0.236),
        "fib_382": high - (diff * 0.382),
        "fib_500": high - (diff * 0.500),
        "fib_618": high - (diff * 0.618),
        "fib_786": high - (diff * 0.786),
        "fib_1": low
    }

def calc_pivot_points(df):
    prev = df.iloc[-2]
    pp = (prev['high'] + prev['low'] + prev['close']) / 3
    r1 = 2 * pp - prev['low']
    s1 = 2 * pp - prev['high']
    r2 = pp + (prev['high'] - prev['low'])
    s2 = pp - (prev['high'] - prev['low'])
    return {"pp": pp, "r1": r1, "r2": r2, "s1": s1, "s2": s2}

def detect_candle_pattern(df):
    c1, c2 = df.iloc[-2], df.iloc[-1]
    patterns = []
    
    if c2['close'] > c2['open'] and c1['close'] < c1['open']:
        if c2['close'] > c1['open'] and c2['open'] < c1['close']:
            patterns.append("Bullish Engulfing")
    
    if c2['close'] < c2['open'] and c1['close'] > c1['open']:
        if c2['close'] < c1['open'] and c2['open'] > c1['close']:
            patterns.append("Bearish Engulfing")
    
    body = abs(c2['close'] - c2['open'])
    wick = c2['high'] - c2['low']
    if wick > 0 and body / wick < 0.1:
        patterns.append("Doji")
    
    lower_wick = min(c2['open'], c2['close']) - c2['low']
    upper_wick = c2['high'] - max(c2['open'], c2['close'])
    if lower_wick > body * 2 and upper_wick < body:
        patterns.append("Hammer")
    
    if upper_wick > body * 2 and lower_wick < body:
        patterns.append("Shooting Star")
    
    return patterns if patterns else ["None"]

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

def get_open_interest(symbol):
    try:
        r = requests.get(f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}", timeout=10)
        return float(r.json()["openInterest"])
    except: return 0.0

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
        scores, titles = [], []
        high_impact = False
        for entry in feed.entries[:5]:
            title = entry.title[:80]
            tl = title.lower()
            if any(kw in tl for kw in HIGH_IMPACT): high_impact = True
            score = analyzer.polarity_scores(tl)['compound']
            bull = sum(1 for kw in BULLISH_KW if kw in tl)
            bear = sum(1 for kw in BEARISH_KW if kw in tl)
            scores.append(score + (bull - bear) * 0.1)
            titles.append(title)
        return {"high_impact": high_impact, "sentiment": sum(scores)/len(scores) if scores else 0, "titles": titles}
    except:
        return {"high_impact": False, "sentiment": 0, "titles": []}

def calculate_confluence_ultimate(df_5m, df_15m, df_1h, fg, whale, sm):
    latest = df_5m.iloc[-1]
    latest_15m = df_15m.iloc[-1]
    latest_1h = df_1h.iloc[-1]
    
    L, S = 0, 0
    sig_l, sig_s = [], []
    
    if latest_1h['ema_50'] > latest_1h['ema_200'] and latest_15m['ema_50'] > latest_15m['ema_200']:
        L += 2; sig_l.append("MTF Bull Aligned")
    elif latest_1h['ema_50'] < latest_1h['ema_200'] and latest_15m['ema_50'] < latest_15m['ema_200']:
        S += 2; sig_s.append("MTF Bear Aligned")
    
    if latest['ema_9'] > latest['ema_21'] > latest['ema_50']:
        L += 1; sig_l.append("EMA Stack Bull")
    elif latest['ema_9'] < latest['ema_21'] < latest['ema_50']:
        S += 1; sig_s.append("EMA Stack Bear")
    
    if latest['close'] > latest['ichi_a'] and latest['close'] > latest['ichi_b']:
        L += 1; sig_l.append("Above Ichi Cloud")
    elif latest['close'] < latest['ichi_a'] and latest['close'] < latest['ichi_b']:
        S += 1; sig_s.append("Below Ichi Cloud")
    
    if latest['close'] > latest['psar']:
        L += 1; sig_l.append("PSAR Bull")
    else:
        S += 1; sig_s.append("PSAR Bear")
    
    if 30 < latest['rsi'] < 50 and latest['rsi'] > df_5m.iloc[-2]['rsi']:
        L += 1; sig_l.append(f"RSI Recovery ({latest['rsi']:.0f})")
    elif 50 < latest['rsi'] < 70 and latest['rsi'] < df_5m.iloc[-2]['rsi']:
        S += 1; sig_s.append(f"RSI Declining ({latest['rsi']:.0f})")
    
    if latest['macd'] > latest['macd_signal'] and latest['macd_hist'] > 0:
        L += 1; sig_l.append("MACD Bull+Hist")
    elif latest['macd'] < latest['macd_signal'] and latest['macd_hist'] < 0:
        S += 1; sig_s.append("MACD Bear+Hist")
    
    if latest['stoch_k'] < 20 and latest['stoch_k'] > latest['stoch_d']:
        L += 1; sig_l.append("Stoch Oversold+Cross")
    elif latest['stoch_k'] > 80 and latest['stoch_k'] < latest['stoch_d']:
        S += 1; sig_s.append("Stoch Overbought+Cross")
    
    if latest['williams_r'] < -80:
        L += 1; sig_l.append("Williams Oversold")
    elif latest['williams_r'] > -20:
        S += 1; sig_s.append("Williams Overbought")
    
    if latest['cci'] < -100:
        L += 1; sig_l.append(f"CCI Oversold ({latest['cci']:.0f})")
    elif latest['cci'] > 100:
        S += 1; sig_s.append(f"CCI Overbought ({latest['cci']:.0f})")
    
    if latest['mfi'] < 20:
        L += 1; sig_l.append("MFI Oversold")
    elif latest['mfi'] > 80:
        S += 1; sig_s.append("MFI Overbought")
    
    if latest['close'] < latest['bb_lower']:
        L += 1; sig_l.append("Below BB Lower")
    elif latest['close'] > latest['bb_upper']:
        S += 1; sig_s.append("Above BB Upper")
    
    if latest['close'] > latest['vwap'] and -0.5 < latest['vwap_dev'] < 1.5:
        L += 1; sig_l.append(f"Above VWAP (+{latest['vwap_dev']:.2f}%)")
    elif latest['close'] < latest['vwap'] and -1.5 < latest['vwap_dev'] < 0.5:
        S += 1; sig_s.append(f"Below VWAP ({latest['vwap_dev']:.2f}%)")
    
    if latest['volume'] > latest['volume_sma'] * 1.5:
        if latest['close'] > latest['open']:
            L += 1; sig_l.append("High Vol + Green")
        else:
            S += 1; sig_s.append("High Vol + Red")
    
    if latest['cmf'] > 0.1:
        L += 1; sig_l.append(f"CMF Positive ({latest['cmf']:.2f})")
    elif latest['cmf'] < -0.1:
        S += 1; sig_s.append(f"CMF Negative ({latest['cmf']:.2f})")
    
    if latest['obv'] > latest['obv_ma']:
        L += 1; sig_l.append("OBV Rising")
    else:
        S += 1; sig_s.append("OBV Falling")
    
    if latest['order_flow_ma'] > 15:
        L += 1; sig_l.append(f"Buy Flow +{latest['order_flow_ma']:.1f}%")
    elif latest['order_flow_ma'] < -15:
        S += 1; sig_s.append(f"Sell Flow {latest['order_flow_ma']:.1f}%")
    
    if latest['adx'] > 25:
        if latest['di_plus'] > latest['di_minus']:
            L += 1; sig_l.append(f"ADX Bull ({latest['adx']:.0f})")
        else:
            S += 1; sig_s.append(f"ADX Bear ({latest['adx']:.0f})")
    
    if whale['pressure'] == "BUY" and (sm['swept_low'] or sm['fvg_bull']):
        L += 2; sig_l.append("WHALE BUY + SMC")
    elif whale['pressure'] == "SELL" and (sm['swept_high'] or sm['fvg_bear']):
        S += 2; sig_s.append("WHALE SELL + SMC")
    elif whale['pressure'] == "BUY":
        L += 1; sig_l.append("Whale Buying")
    elif whale['pressure'] == "SELL":
        S += 1; sig_s.append("Whale Selling")
    
    if "UPTREND" in sm['structure']:
        L += 1; sig_l.append("Structure: HH+HL")
    elif "DOWNTREND" in sm['structure']:
        S += 1; sig_s.append("Structure: LL+LH")
    
    if fg['value'] < 25:
        L += 1; sig_l.append(f"Extreme Fear ({fg['value']})")
    elif fg['value'] > 75:
        S += 1; sig_s.append(f"Extreme Greed ({fg['value']})")
    
    return {"long": L, "short": S, "sig_long": sig_l, "sig_short": sig_s}

def create_setup(direction, latest, atr, sm, fib, pivot):
    entry = latest['close']
    if direction == "LONG":
        sl_candidates = [entry - (atr * CONFIG["atr_sl_mult"])]
        if sm['bull_ob']: sl_candidates.append(sm['bull_ob'] * 0.998)
        if pivot['s1'] < entry: sl_candidates.append(pivot['s1'])
        sl = min(sl_candidates)
        tp1 = entry + (atr * 2.0)
        tp2 = entry + (atr * CONFIG["atr_tp_mult"])
        tp3 = min(pivot['r2'], entry + (atr * 5.0))
    else:
        sl_candidates = [entry + (atr * CONFIG["atr_sl_mult"])]
        if sm['bear_ob']: sl_candidates.append(sm['bear_ob'] * 1.002)
        if pivot['r1'] > entry: sl_candidates.append(pivot['r1'])
        sl = max(sl_candidates)
        tp1 = entry - (atr * 2.0)
        tp2 = entry - (atr * CONFIG["atr_tp_mult"])
        tp3 = max(pivot['s2'], entry - (atr * 5.0))
    risk = abs(entry - sl)
    reward = abs(tp2 - entry)
    return {"entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3, "rr": reward/risk if risk > 0 else 0}

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

def analyze_and_send(symbol):
    try:
        print(f">>> Analyzing {symbol}...")
        
        df_5m = calc_all_indicators(get_klines(symbol, "5m", 500))
        df_15m = calc_all_indicators(get_klines(symbol, "15m", 500))
        df_1h = calc_all_indicators(get_klines(symbol, "1h", 500))
        
        latest = df_5m.iloc[-1]
        atr = latest['atr']
        
        fg = get_fear_greed()
        funding = get_funding(symbol)
        ls = get_long_short(symbol)
        oi = get_open_interest(symbol)
        whale = get_whale(symbol)
        news = check_news()
        sm = detect_smart_money(df_5m)
        fib = calc_fibonacci(df_5m)
        pivot = calc_pivot_points(df_1h)
        patterns = detect_candle_pattern(df_5m)
        
        conf = calculate_confluence_ultimate(df_5m, df_15m, df_1h, fg, whale, sm)
        
        support = df_5m["low"].tail(50).min()
        resistance = df_5m["high"].tail(50).max()
        trend_1h = "BULLISH" if df_1h.iloc[-1]['ema_50'] > df_1h.iloc[-1]['ema_200'] else "BEARISH"
        trend_5m = "BULLISH" if latest['ema_9'] > latest['ema_21'] else "BEARISH"
        
        if conf['long'] > conf['short']:
            direction = "LONG"
            setup = create_setup("LONG", latest, atr, sm, fib, pivot)
            signals = conf['sig_long']
            score = conf['long']
        else:
            direction = "SHORT"
            setup = create_setup("SHORT", latest, atr, sm, fib, pivot)
            signals = conf['sig_short']
            score = conf['short']
        
        grade = "A+" if score >= 14 else "A" if score >= 10 else "B" if score >= 7 else "C"
        priority = "high" if score >= 12 else "default"
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        title = f"{direction} {symbol} | ${setup['entry']:,.0f} | {grade} {score}/20"
        
        body = f"""ULTIMATE TRADING SIGNAL v9.0
{timestamp} | {symbol} 5M

=== SIGNAL QUALITY ===
Direction: {direction}
Grade: {grade}
Confluence: {score}/20
Priority: {'HIGH' if score >= 12 else 'NORMAL'}

=== TRADE SETUP ===
Entry: ${setup['entry']:,.2f}
Stop Loss: ${setup['sl']:,.2f}
TP1 (50%): ${setup['tp1']:,.2f}
TP2 (30%): ${setup['tp2']:,.2f}
TP3 (20%): ${setup['tp3']:,.2f}
Risk/Reward: 1:{setup['rr']:.2f}

=== PRICE ACTION ===
Price: ${latest['close']:,.2f}
Support: ${support:,.2f}
Resistance: ${resistance:,.2f}
Trend 1H: {trend_1h}
Trend 5M: {trend_5m}
Market Structure: {sm['structure']}
Candle Pattern: {', '.join(patterns)}

=== TREND INDICATORS ===
EMA 9: ${latest['ema_9']:,.2f}
EMA 21: ${latest['ema_21']:,.2f}
EMA 50: ${latest['ema_50']:,.2f}
EMA 100: ${latest['ema_100']:,.2f}
EMA 200: ${latest['ema_200']:,.2f}
SMA 50: ${latest['sma_50']:,.2f}
SMA 200: ${latest['sma_200']:,.2f}
Ichimoku A: ${latest['ichi_a']:,.2f}
Ichimoku B: ${latest['ichi_b']:,.2f}
PSAR: ${latest['psar']:,.2f}
Donchian U/L: ${latest['donch_upper']:,.2f} / ${latest['donch_lower']:,.2f}

=== MOMENTUM ===
RSI 14: {latest['rsi']:.2f}
RSI 7: {latest['rsi_fast']:.2f}
MACD: {latest['macd']:.6f}
MACD Signal: {latest['macd_signal']:.6f}
MACD Hist: {latest['macd_hist']:.6f}
Stoch K/D: {latest['stoch_k']:.2f} / {latest['stoch_d']:.2f}
Williams %R: {latest['williams_r']:.2f}
CCI: {latest['cci']:.2f}
ROC: {latest['roc']:.2f}%
Awesome: {latest['awesome']:.2f}

=== VOLATILITY ===
BB Upper: ${latest['bb_upper']:,.2f}
BB Lower: ${latest['bb_lower']:,.2f}
BB Width: {latest['bb_width']:.2f}%
Keltner U/L: ${latest['kc_upper']:,.2f} / ${latest['kc_lower']:,.2f}
ATR: ${atr:.2f}

=== VOLUME ANALYSIS ===
Volume: {latest['volume']:,.0f}
Volume MA: {latest['volume_sma']:,.0f}
OBV: {latest['obv']:,.0f}
CMF: {latest['cmf']:.3f}
MFI: {latest['mfi']:.2f}
Order Flow: {latest['order_flow_ma']:.2f}%

=== VWAP ===
VWAP: ${latest['vwap']:,.2f}
VWAP Deviation: {latest['vwap_dev']:.2f}%

=== TREND STRENGTH ===
ADX: {latest['adx']:.2f}
DI+: {latest['di_plus']:.2f}
DI-: {latest['di_minus']:.2f}

=== SMART MONEY ===
Market Structure: {sm['structure']}
Bullish OB: ${sm['bull_ob']:,.2f}
Bearish OB: ${sm['bear_ob']:,.2f}
Swept High: {'YES' if sm['swept_high'] else 'NO'}
Swept Low: {'YES' if sm['swept_low'] else 'NO'}
FVG Bullish: {'YES' if sm['fvg_bull'] else 'NO'}
FVG Bearish: {'YES' if sm['fvg_bear'] else 'NO'}

=== WHALE ACTIVITY ===
Pressure: {whale['pressure']}
Buy Volume: {whale['buy']:.2f}
Sell Volume: {whale['sell']:.2f}

=== FIBONACCI LEVELS ===
Fib 0.236: ${fib['fib_236']:,.2f}
Fib 0.382: ${fib['fib_382']:,.2f}
Fib 0.500: ${fib['fib_500']:,.2f}
Fib 0.618: ${fib['fib_618']:,.2f}
Fib 0.786: ${fib['fib_786']:,.2f}

=== PIVOT POINTS (1H) ===
R2: ${pivot['r2']:,.2f}
R1: ${pivot['r1']:,.2f}
PP: ${pivot['pp']:,.2f}
S1: ${pivot['s1']:,.2f}
S2: ${pivot['s2']:,.2f}

=== FUTURES DATA ===
Funding Rate: {funding:.4f}%
Open Interest: {oi:,.0f}
Long/Short Ratio: {ls['ratio']:.2f}
Long: {ls['long']:.1f}% | Short: {ls['short']:.1f}%

=== SENTIMENT ===
Fear & Greed: {fg['value']}/100 ({fg['class']})
News Sentiment: {news['sentiment']:.2f}
High Impact News: {'YES' if news['high_impact'] else 'NO'}

=== ACTIVE SIGNALS ({len(signals)}) ===
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

Sebagai expert trader profesional institusional dengan pengalaman 15 tahun di hedge fund, tolong analisis data trading {symbol} 5M scalping di atas dengan komprehensif:

1. VALIDASI SETUP {direction}
   - Apakah setup ini valid dengan {score}/20 confluence?
   - Tingkat kepercayaan setup (0-100%)?
   - Identifikasi red flags atau konflik antar indikator

2. ANALISIS TEKNIKAL MULTI-INDIKATOR
   - Konfirmasi trend (EMA/SMA/Ichimoku/PSAR)
   - Momentum (RSI, MACD, Stoch, Williams, CCI, MFI)
   - Volatility (BB, Keltner, ATR) - apakah trending atau ranging?
   - Volume (OBV, CMF, MFI) - apakah mendukung direction?
   - VWAP & Order Flow - institutional bias?
   - ADX & DI - kekuatan trend?

3. SMART MONEY ANALYSIS
   - Order Block behavior
   - Liquidity Sweep interpretation
   - Fair Value Gap implication
   - Market Structure validity
   - Whale activity correlation

4. KEY LEVELS
   - Fibonacci level yang paling relevan sebagai target/support
   - Pivot Points untuk scalping
   - Support/Resistance paling kuat

5. SENTIMENT & MACRO
   - Fear & Greed impact
   - Funding rate signal (kontrarian/trending)
   - Long/Short ratio interpretation
   - News impact ke harga

6. REFINEMENT ENTRY
   - Apakah entry ${setup['entry']:,.2f} optimal?
   - Rekomendasi entry lebih baik
   - Timing: sekarang/pullback/breakout

7. RISK MANAGEMENT
   - Validasi SL di ${setup['sl']:,.2f}
   - TP realistis: ${setup['tp1']:,.2f} / ${setup['tp2']:,.2f} / ${setup['tp3']:,.2f}
   - Position sizing saran (% modal)
   - Risk per trade max

8. SKENARIO
   - Skenario Bullish: target & trigger
   - Skenario Bearish: invalidasi & exit
   - Skenario Sideways: action plan

9. KEPUTUSAN FINAL
   - EKSEKUSI SEKARANG / TUNGGU KONFIRMASI / SKIP
   - Confidence level 0-100%
   - Time frame: scalp 15m / 1-2 jam / swing
   - Alasan 3 poin utama

Berikan analisis dalam format terstruktur dan actionable.
"""
        
        send_ntfy(title, body, priority)
        print(f"   OK {direction} @ ${setup['entry']:,.0f} | {grade} ({score}/20)")
        
    except Exception as e:
        print(f"   Error: {e}")

if __name__ == "__main__":
    print("="*60)
    print("ULTIMATE TRADING BOT v9.0 - MULTI COIN")
    print("="*60)
    print(f"Coins: {', '.join(CONFIG['symbols'])}")
    print("Signal delivery: Ntfy HP")
    print("Timeframe: 5M scalping")
    print("\nVPN must be active!")
    print("Press Control+C to stop\n")
    
    send_ntfy("ULTIMATE BOT STARTED", f"Ultimate Trading Bot v9.0 is LIVE\nCoins: {', '.join(CONFIG['symbols'])}\n20 indicators aktif")
    
    while True:
        try:
            for symbol in CONFIG["symbols"]:
                analyze_and_send(symbol)
                time.sleep(3)
            print("\nWait 5 minutes for next scan...\n")
            time.sleep(300)
        except KeyboardInterrupt:
            send_ntfy("BOT STOPPED", "Bot telah dimatikan")
            print("\nStopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(30)
