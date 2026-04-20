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

def send_data():
    try:
        df = calc(get_data())
        latest = df.iloc[-1]
        
        price = latest['close']
        rsi = latest['rsi']
        macd = latest['macd']
        macd_signal = latest['macd_signal']
        bb_upper = latest['bb_upper']
        bb_lower = latest['bb_lower']
        ema_50 = latest['ema_50']
        ema_200 = latest['ema_200']
        atr = latest['atr']
        volume = latest['volume']
        volume_sma = latest['volume_sma']
        
        support = df["low"].tail(50).min()
        resistance = df["high"].tail(50).max()
        
        trend = "BULLISH 📈" if ema_50 > ema_200 else "BEARISH 📉"
        signal = "LONG" if ema_50 > ema_200 else "SHORT"
        
        # Entry & SL & TP
        entry = price
        sl = entry - (atr * 1.5)
        tp1 = entry + (atr * 2.0)
        tp2 = entry + (atr * 4.5)
        rr = (tp2 - entry) / (entry - sl) if entry > sl else 0
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"\n{'='*60}")
        print(f"🎯 SIGNAL - {timestamp}")
        print(f"{'='*60}")
        print(f"\n💰 Price: ${price:,.2f}")
        print(f"📊 Trend: {trend}")
        print(f"📈 Support: ${support:,.2f} | Resistance: ${resistance:,.2f}")
        
        print(f"\n📌 INDICATORS:")
        print(f"   RSI: {rsi:.0f}")
        print(f"   MACD: {macd:.6f} (Signal: {macd_signal:.6f})")
        print(f"   EMA50: ${ema_50:,.2f} | EMA200: ${ema_200:,.2f}")
        print(f"   BB Upper: ${bb_upper:,.2f} | Lower: ${bb_lower:,.2f}")
        print(f"   ATR: ${atr:,.2f}")
        print(f"   Volume: {volume:,.0f} (Avg: {volume_sma:,.0f})")
        
        print(f"\n{'='*60}")
        print(f"📋 COPY BAWAH INI KE CLAUDE PRO:")
        print(f"{'='*60}\n")
        
        copy_text = f"""🎯 TRADING SIGNAL BTCUSDT
📅 {timestamp}

💰 Harga Sekarang: ${price:,.2f}
📈 Trend: {trend}
🎯 Support: ${support:,.2f}
🎯 Resistance: ${resistance:,.2f}

📊 TECHNICAL INDICATORS:
- RSI 14: {rsi:.0f}
- MACD: {macd:.6f}
- MACD Signal: {macd_signal:.6f}
- EMA 50: ${ema_50:,.2f}
- EMA 200: ${ema_200:,.2f}
- Bollinger Upper: ${bb_upper:,.2f}
- Bollinger Lower: ${bb_lower:,.2f}
- ATR: ${atr:,.2f}
- Volume: {volume:,.0f}
- Volume MA: {volume_sma:,.0f}

📋 TRADE SETUP:
Entry: ${entry:,.2f}
Stop Loss: ${sl:,.2f}
Take Profit 1: ${tp1:,.2f}
Take Profit 2: ${tp2:,.2f}
Risk Reward: 1:{rr:.2f}

🎯 SINYAL: {signal}

Mohon analisis dan berikan keputusan trading berdasarkan data di atas."""
        
        print(copy_text)
        print(f"\n{'='*60}\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🤖 ULTRA SIMPLE BOT")
    print("📊 Setiap 5 menit kasih signal")
    print("📱 Copy ke Claude Pro di HP")
    print("⚠️  VPN harus aktif!")
    print("Tekan Control+C untuk stop\n")
    
    while True:
        try:
            send_data()
            print("⏳ Tunggu 5 menit untuk signal berikutnya...\n")
            time.sleep(300)
        except KeyboardInterrupt:
            print("\n👋 Bot stopped.")
            break
        except Exception as e:
            print(f"❌ Error di main loop: {e}")
            time.sleep(30)
