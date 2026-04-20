import os
from binance.client import Client
from dotenv import load_dotenv
import pandas as pd
import ta
from datetime import datetime

load_dotenv()
api_key    = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_SECRET_KEY")
client = Client(api_key, api_secret)

def get_data(symbol="BTCUSDT", limit=200):
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
    df["volume_sma"] = df["volume"].rolling(window=20).mean()
    
    return df.dropna()

def analyze(symbol="BTCUSDT"):
    df = get_data(symbol, limit=200)
    df = calc_indicators(df)
    latest = df.iloc[-1]
    
    trend_long = "BULLISH" if latest['ema_50'] > latest['ema_200'] else "BEARISH"
    trend_short = "BULLISH" if latest['ema_12'] > latest['ema_26'] else "BEARISH"
    
    print("\n" + "="*70)
    print(f"🔍 TRADING SIGNAL - {symbol} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    print("\n📋 COPY SEMUA INI KE CLAUDE PRO:\n")
    print(f"""
Analisis Technical Trading {symbol}:

=== HARGA & TREND ===
Harga Sekarang      : ${latest['close']:.2f}
EMA 12              : ${latest['ema_12']:.2f}
EMA 26              : ${latest['ema_26']:.2f}
EMA 50              : ${latest['ema_50']:.2f}
EMA 200             : ${latest['ema_200']:.2f}
Trend Jangka Panjang: {trend_long}
Trend Jangka Pendek : {trend_short}

=== MOMENTUM ===
RSI 14              : {latest['rsi']:.2f}
RSI 7               : {latest['rsi_fast']:.2f}
Stochastic K        : {latest['stoch_k']:.2f}
Stochastic D        : {latest['stoch_d']:.2f}
MACD                : {latest['macd']:.6f}
MACD Signal         : {latest['macd_signal']:.6f}
MACD Histogram       : {latest['macd_hist']:.6f}

=== VOLATILITY ===
Bollinger Upper     : ${latest['bb_upper']:.2f}
Bollinger Mid       : ${latest['bb_mid']:.2f}
Bollinger Lower     : ${latest['bb_lower']:.2f}
ATR (Volatility)    : ${latest['atr']:.2f}

=== VOLUME ===
Volume Sekarang     : {latest['volume']:,.2f}
Volume MA           : {latest['volume_sma']:,.2f}
Status              : {'TINGGI' if latest['volume'] > latest['volume_sma'] else 'RENDAH'}

Berikan keputusan trading:
1. Signal: LONG / SHORT / WAIT
2. Confidence: 0-100%
3. Entry Price
4. Stop Loss
5. Take Profit
6. Risk Level: LOW / MEDIUM / HIGH
""")
    print("="*70)

if __name__ == "__main__":
    import time
    print("🤖 Trading Signal Bot Started")
    print("VPN harus aktif!")
    print("Tekan Control + C untuk stop\n")
    
    while True:
        try:
            analyze("BTCUSDT")
            print("\nTekan Enter untuk update sekarang, atau tunggu 5 menit...")
            time.sleep(300)
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(10)
