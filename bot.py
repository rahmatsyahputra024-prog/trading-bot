import requests
import pandas as pd
import pandas_ta as ta
import google.generativeai as genai
import time
import os

# === KONFIGURASI ===
GEMINI_API_KEY = "PASTE_API_KEY_GEMINI_ANDA" # Ambil di aistudio.google.com
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def ambil_data():
    # Mengambil data harga, volume, dan indikator
    url_hist = "https://min-api.cryptocompare.com/data/v2/histominute?fsym=SOL&tsyms=USD&limit=50"
    url_realtime = "https://min-api.cryptocompare.com/data/pricemultifull?fsyms=SOL&tsyms=USD"
    url_news = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"

    try:
        # 1. Olah Indikator (RSI)
        res_hist = requests.get(url_hist).json()
        df = pd.DataFrame(res_hist['Data']['Data'])
        df['RSI'] = ta.rsi(df['close'], length=14)
        rsi_now = df['RSI'].iloc[-1]

        # 2. Olah Realtime Data & Volume
        res_real = requests.get(url_realtime).json()['RAW']['SOL']['USD']
        harga = res_real['PRICE']
        vol_24j = res_real['VOLUME24HOUR']

        # 3. Olah Berita
        res_news = requests.get(url_news).json()['Data'][0]
        berita = res_news['title']

        return {
            "harga": harga,
            "rsi": round(rsi_now, 2),
            "volume": f"${vol_24j:,.0f}",
            "berita": berita
        }
    except Exception as e:
        print(f"Gagal ambil data: {e}")
        return None

def minta_analisis_gemini(data):
    prompt = f"""
    Sebagai pakar trading, analisis data SOL/USD ini:
    - Harga: ${data['harga']}
    - RSI (14): {data['rsi']}
    - Volume 24J: {data['volume']}
    - Berita Terbaru: {data['berita']}
    
    Tugasmu:
    Berikan instruksi (BUY/SELL/WAIT) beserta alasan teknikal & sentimen beritanya secara singkat dan tajam.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini Error: {e}"

if __name__ == "__main__":
    print("🔥 BOT ANALIS REALTIME DIMULAI (CTRL+C untuk berhenti) 🔥")
    while True:
        os.system('clear') # Biar tampilan terminal bersih tiap update
        data_pasar = ambil_data()
        
        if data_pasar:
            print(f"📊 DATA PASAR | Harga: ${data_pasar['harga']} | RSI: {data_pasar['rsi']}")
            print(f"📰 BERITA: {data_pasar['berita'][:80]}...")
            print("\n🤖 ANALISIS GEMINI:")
            print(minta_analisis_gemini(data_pasar))
            print("\n" + "="*50)
        
        print("\nMenunggu update 1 menit berikutnya...")
        time.sleep(60)