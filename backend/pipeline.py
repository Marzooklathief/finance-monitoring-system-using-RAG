import pathway as pw
import yfinance as yf
from datetime import datetime

# 🛠️ Create Pathway Table for Real-time Stock Prices
class StockData(pw.Schema):
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

# 🎯 Fetch Stock Data in Real-time
def fetch_stock_data(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period="1d", interval="1m")
    latest = data.iloc[-1]
    return {
        "symbol": symbol,
        "timestamp": datetime.utcnow(),
        "open": latest["Open"],
        "high": latest["High"],
        "low": latest["Low"],
        "close": latest["Close"],
        "volume": int(latest["Volume"]),
    }

# 📡 Pathway Data Stream
stock_stream = pw.io.periodic(lambda: fetch_stock_data("TSLA"), interval_s=60).with_schema(StockData)

# 🚨 Detect Anomalies (e.g., sudden price drops)
def detect_risk(data):
    return data.select(data.symbol, data.close, risk_alert=(data.close < data.open * 0.97))

risk_alerts = detect_risk(stock_stream)

# 📤 Output Risk Alerts
pw.io.stdout(risk_alerts)

pw.run()
