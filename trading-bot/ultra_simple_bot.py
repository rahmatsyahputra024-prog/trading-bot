import os
import time
import requests
from datetime import datetime
from binance.client import Client
from dotenv import load_dotenv
import pandas as pd
import ta

load_dotenv()
client = Client(os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_SECRET_KEY"))

NTFY_URL = "https://ntfy.sh/Mamat-trading"

def get_data(symbol="BTCUSDT"):
    k = client.get_klines(symbol=symbol, interval="5m", limit=500)
    df = pd.DataFrame(k, columns=["time","open","high","low","close","volume","close_time","quote_vol","trades","taker_buy_base","taker_buy_quote","ignore"])
    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
    return df

def calc(df):
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(df["close"], window=20)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["ema_50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df["ema_200"] = ta.trend.EMAIndicator(df["close"], window=200).ema_indicator()
    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    df["volume_sma"] = df["volume"].rolling(window=20).mean()
    return df.dropna()

def send_signal():
    try:
        df = calc(get_data())
        latest = df.iloc[-1]
        
        price = latest['close']
        rsi = latest['rsi']
        macd = latest['macd']
        ema_50 = latest['ema_50']
        ema_200 = latest['ema_200']
        atr = latest['atr']
        
        support = df["low"].tail(50).min()
        resistance = df["high"].tail(50).max()
        
        trend = "BULLISH" if ema_50 > ema_200 else "BEARISH"
        signal = "LONG" if ema_50 > ema_200 else "SHORT"
        
        entry = price
        sl = entry - (atr * 1.5)
        tp1 = entry + (atr * 2.0)
        tp2 = entry + (atr * 4.5)
        rr = (tp2 - entry) / (entry - sl) if entry > sl else 0
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        title = f"{signal} | ${price:,.0f} | RSI {rsi:.0f}"
        
        body = f"""BTCUSDT {timestamp}

Price: ${price:,.2f}
Trend: {trend}
Support: ${support:,.2f}
Resistance: ${resistance:,.2f}

RSI: {rsi:.0f}
MACD: {macd:.6f}
EMA50: ${ema_50:,.2f}
EMA200: ${ema_200:,.2f}

Entry: ${entry:,.2f}
SL: ${sl:,.2f}
TP1: ${tp1:,.2f}
TP2: ${tp2:,.2f}
RR: 1:{rr:.2f}

Signal: {signal}"""
        
        requests.post(NTFY_URL, data=body.encode('utf-8'), headers={"Title": title})
        
        print(f"OK [{timestamp}] Signal sent!")
        print(f"   {title}\n")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("BOT TRADING - MAMAT-TRADING")
    print("Signal setiap 5 menit")
    print("Tekan Control+C untuk stop\n")
    
    while True:
        try:
            send_signal()
            print("Tunggu 5 menit...\n")
            time.sleep(300)
        except KeyboardInterrupt:
            print("\nStop.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(30)
