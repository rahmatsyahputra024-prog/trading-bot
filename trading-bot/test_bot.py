import os
from dotenv import load_dotenv
from binance.client import Client

load_dotenv()
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_SECRET_KEY")

print(f"✅ API Key found: {api_key[:10]}...")
print(f"✅ API Secret found: {api_secret[:10]}...")

client = Client(api_key, api_secret)
ticker = client.get_symbol_ticker(symbol="BTCUSDT")
print(f"✅ BTC Price: ${ticker['price']}")
