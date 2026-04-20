import os
import time
import requests
from datetime import datetime
from binance.client import Client
from dotenv import load_dotenv
import pandas as pd
import ta

load_dotenv()
api_key    = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_SECRET_KEY")
client = Client(api_key, api_secret)

# ==================== DATA COLLECTION ====================

def get_market_data(symbol="BTCUSDT", limit=200):
    klines = client.get_klines(symbol=symbol, interval="5m", limit=limit)
    df = pd.DataFrame(klines, columns=[
        "time","open","high","low","close","volume",
        "close_time","quote_vol","trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ])
    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
    return df

def get_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        data = r.json()["data"][0]
        return {
            "value": int(data["value"]),
            "classification": data["value_classification"]
        }
    except:
        return {"value": 50, "classification": "Unknown"}

def get_funding_rate(symbol="BTCUSDT"):
    try:
        r = requests.get(
            f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}",
            timeout=10
        )
        data = r.json()
        return float(data["lastFundingRate"]) * 100
    except:
        return 0.0

def get_open_interest(symbol="BTCUSDT"):
    try:
        r = requests.get(
            f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}",
            timeout=10
        )
        return float(r.json()["openInterest"])
    except:
        return 0.0

def get_long_short_ratio(symbol="BTCUSDT"):
    try:
        r = requests.get(
            f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={symbol}&period=1h&limit=1",
            timeout=10
        )
        data = r.json()[0]
        return {
            "long": float(data["longAccount"]) * 100,
            "short": float(data["shortAccount"]) * 100,
            "ratio": float(data["longShortRatio"])
        }
    except:
        return {"long": 50, "short": 50, "ratio": 1.0}

def get_whale_trades(symbol="BTCUSDT", min_qty=5):
    """Deteksi whale trades >5 BTC (~$375k+)"""
    try:
        r = requests.get(
            f"https://api.binance.com/api/v3/trades?symbol={symbol}&limit=500",
            timeout=10
        )
        trades = r.json()
        whale_buys = 0
        whale_sells = 0
        whale_volume = 0
        
        for t in trades:
            qty = float(t["qty"])
            if qty >= min_qty:
                whale_volume += qty
                if t["isBuyerMaker"]:
                    whale_sells += qty
                else:
                    whale_buys += qty
        
        return {
            "buy_volume": whale_buys,
            "sell_volume": whale_sells,
            "total_volume": whale_volume,
            "pressure": "BUY" if whale_buys > whale_sells else "SELL"
        }
    except:
        return {"buy_volume": 0, "sell_volume": 0, "total_volume": 0, "pressure": "NEUTRAL"}

def get_volume_profile(df, bins=10):
    """Volume Profile - area harga dengan volume tertinggi"""
    price_range = df["close"].max() - df["close"].min()
    bin_size = price_range / bins
    
    profile = {}
    for i in range(bins):
        lower = df["close"].min() + (i * bin_size)
        upper = lower + bin_size
        mask = (df["close"] >= lower) & (df["close"] < upper)
        profile[f"${lower:.0f}-${upper:.0f}"] = df.loc[mask, "volume"].sum()
    
    poc = max(profile, key=profile.get)
    return poc, profile[poc]

# ==================== INDICATORS ====================

def calc_indicators(df):
    df["rsi"]       = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["rsi_fast"]  = ta.momentum.RSIIndicator(df["close"], window=7).rsi()
    
    macd = ta.trend.MACD(df["close"])
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"]   = macd.macd_diff()
    
    bb = ta.volatility.BollingerBands(df["close"], window=20)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"]   = bb.bollinger_mavg()
    
    stoch = ta.momentum.StochRSIIndicator(df["close"], window=14)
    df["stoch_k"] = stoch.stochrsi_k() * 100
    df["stoch_d"] = stoch.stochrsi_d() * 100
    
    df["ema_12"]  = ta.trend.EMAIndicator(df["close"], window=12).ema_indicator()
    df["ema_26"]  = ta.trend.EMAIndicator(df["close"], window=26).ema_indicator()
    df["ema_50"]  = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df["ema_200"] = ta.trend.EMAIndicator(df["close"], window=200).ema_indicator()
    
    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    df["adx"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14).adx()
    df["volume_sma"] = df["volume"].rolling(window=20).mean()
    
    return df.dropna()

# ==================== ANALYSIS ====================

def analyze_all(symbol="BTCUSDT"):
    print("\n🔄 Mengumpulkan data...")
    
    df = get_market_data(symbol, limit=200)
    df = calc_indicators(df)
    latest = df.iloc[-1]
    
    fg            = get_fear_greed()
    funding       = get_funding_rate(symbol)
    oi            = get_open_interest(symbol)
    ls_ratio      = get_long_short_ratio(symbol)
    whale         = get_whale_trades(symbol)
    poc, poc_vol  = get_volume_profile(df)
    
    support       = df["low"].tail(50).min()
    resistance    = df["high"].tail(50).max()
    trend_long    = "BULLISH 📈" if latest['ema_50'] > latest['ema_200'] else "BEARISH 📉"
    trend_short   = "BULLISH 📈" if latest['ema_12'] > latest['ema_26'] else "BEARISH 📉"
    
    print("\n" + "="*75)
    print(f"🎯 PRO TRADING SIGNAL - {symbol} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*75)
    print("\n📋 COPY SEMUA INI KE CLAUDE PRO:\n")
    print(f"""
Analisis LENGKAP Trading {symbol} dengan data institusional:

========== 1. PRICE & TREND ==========
Harga Sekarang         : ${latest['close']:.2f}
Support                : ${support:.2f}
Resistance             : ${resistance:.2f}
EMA 12/26              : ${latest['ema_12']:.2f} / ${latest['ema_26']:.2f}
EMA 50/200             : ${latest['ema_50']:.2f} / ${latest['ema_200']:.2f}
Trend Jangka Panjang   : {trend_long}
Trend Jangka Pendek    : {trend_short}

========== 2. MOMENTUM ==========
RSI 14 / RSI 7         : {latest['rsi']:.2f} / {latest['rsi_fast']:.2f}
Stochastic K/D         : {latest['stoch_k']:.2f} / {latest['stoch_d']:.2f}
MACD / Signal / Hist   : {latest['macd']:.4f} / {latest['macd_signal']:.4f} / {latest['macd_hist']:.4f}
ADX (Trend Strength)   : {latest['adx']:.2f} {'STRONG' if latest['adx'] > 25 else 'WEAK'}

========== 3. VOLATILITY ==========
Bollinger Upper/Lower  : ${latest['bb_upper']:.2f} / ${latest['bb_lower']:.2f}
Bollinger Mid          : ${latest['bb_mid']:.2f}
ATR (Volatility)       : ${latest['atr']:.2f}

========== 4. VOLUME ==========
Volume Sekarang        : {latest['volume']:,.2f}
Volume MA (20)         : {latest['volume_sma']:,.2f}
Status                 : {'TINGGI 🔥' if latest['volume'] > latest['volume_sma'] else 'RENDAH 💤'}
Point of Control (POC) : {poc} (volume tertinggi: {poc_vol:,.0f})

========== 5. SENTIMEN PASAR ==========
Fear & Greed Index     : {fg['value']}/100 ({fg['classification']})
  → {'EXTREME FEAR - potensi bottom' if fg['value'] < 25 else 'EXTREME GREED - potensi top' if fg['value'] > 75 else 'NEUTRAL'}

========== 6. FUTURES DATA ==========
Funding Rate           : {funding:.4f}%
  → {'LONG dominan (bearish signal)' if funding > 0.01 else 'SHORT dominan (bullish signal)' if funding < -0.01 else 'NETRAL'}
Open Interest          : {oi:,.0f} kontrak
Long/Short Ratio       : {ls_ratio['ratio']:.2f}
  → Long: {ls_ratio['long']:.1f}% | Short: {ls_ratio['short']:.1f}%

========== 7. WHALE MOVEMENT ==========
Whale Buy Volume       : {whale['buy_volume']:.2f} {symbol.replace('USDT','')}
Whale Sell Volume      : {whale['sell_volume']:.2f} {symbol.replace('USDT','')}
Whale Total Volume     : {whale['total_volume']:.2f} {symbol.replace('USDT','')}
Whale Pressure         : {whale['pressure']}
  → Whale sedang {'AKUMULASI' if whale['pressure'] == 'BUY' else 'DISTRIBUSI' if whale['pressure'] == 'SELL' else 'MENUNGGU'}

=====================================================
Berikan analisis & keputusan trading:

1. Signal: LONG / SHORT / WAIT
2. Confidence: 0-100%
3. Entry Price
4. Stop Loss
5. Take Profit (TP1, TP2, TP3)
6. Risk Level: LOW / MEDIUM / HIGH
7. Risk/Reward Ratio
8. Time Frame Trading: SCALP / DAYTRADE / SWING
9. Ringkasan kondisi pasar (sentimen + whale + futures)
10. Strategi backup jika market reversal
""")
    print("="*75)

# ==================== MAIN LOOP ====================

if __name__ == "__main__":
    print("🤖 PRO TRADING BOT v2.0")
    print("Data: Binance + Fear&Greed + Whale + Funding Rate + OI + L/S Ratio")
    print("Pastikan VPN aktif!")
    print("Tekan Control + C untuk stop\n")
    
    while True:
        try:
            analyze_all("BTCUSDT")
            print("\n⏳ Update otomatis 5 menit. Tekan Control+C untuk stop.")
            time.sleep(300)
        except KeyboardInterrupt:
            print("\n\n👋 Bot stopped.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(10)
