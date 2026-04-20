from binance.client import Client
from dotenv import load_dotenv
import os
import pandas as pd
import ta

load_dotenv()
api_key    = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_SECRET_KEY")

client = Client(api_key, api_secret)

def analisis_btc():
    print("Mengambil data BTC...")

    klines = client.futures_klines(
        symbol   = "BTCUSDT",
        interval = "1h",
        limit    = 100
    )

    df = pd.DataFrame(klines, columns=[
        "time","open","high","low","close","volume",
        "close_time","quote_vol","trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ])

    df[["open","high","low","close","volume"]] = \
        df[["open","high","low","close","volume"]].astype(float)

    df["rsi"]         = ta.momentum.RSIIndicator(df["close"], 
window=14).rsi()
    df["ema20"]       = ta.trend.EMAIndicator(df["close"], 
window=20).ema_indicator()
    df["ema50"]       = ta.trend.EMAIndicator(df["close"], 
window=50).ema_indicator()

    macd              = ta.trend.MACD(df["close"])
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    df     = df.dropna()
    latest = df.iloc[-1]

    print("\n" + "="*50)
    print("COPY PASTE KE CLAUDE PRO:")
    print("="*50)
    print(f"""
Tolong analisis data trading BTC berikut dan berikan sinyal BUY, SELL, 
atau HOLD:

Harga BTC sekarang : ${latest['close']:,.2f}
RSI (14)           : {latest['rsi']:.2f}
EMA 20             : ${latest['ema20']:,.2f}
EMA 50             : ${latest['ema50']:,.2f}
MACD               : {latest['macd']:.4f}
MACD Signal        : {latest['macd_signal']:.4f}
Tren               : {'BULLISH' if latest['ema20'] > latest['ema50'] else 
'BEARISH'}

Berikan:
1. Sinyal: BUY / SELL / HOLD
2. Alasan singkat
3. Entry price
4. Stop loss
5. Take profit
""")
    print("="*50)

analisis_btc()
