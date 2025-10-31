# STOCKMARKET-SUJALPATEL-
🇮🇳 Indian Stock Predictor

A web application that predicts short-term trends of Indian NSE stocks using real-time market data, built with Flask, yFinance, Chart.js, and Tailwind CSS.

This project provides an interactive dashboard where users can:

View live stock details (price, sector, exchange, summary)

Visualize historical trends and 7-day forecast

Compare multiple stocks

Add stocks to a personal watchlist

View related news & basic sentiment

Export charts and data for analysis

🚀 Features

✅ Search any Indian stock by ticker (e.g., RELIANCE.NS, TCS.NS)
✅ Displays stock details like price, exchange, sector, and business summary
✅ Fetches historical data and predicts the next 7 days using trend extrapolation
✅ Interactive Chart.js graph (historical + forecast with toggle & indicators)
✅ Watchlist saved locally in the browser
✅ Live price updates (auto-refresh)
✅ Comparison mode to visualize multiple stocks together
✅ Currency toggle (INR / USD)
✅ Download chart or export CSV
✅ Sentiment & news preview (with placeholders for News API)
✅ Fully responsive, clean dark UI built with Tailwind CSS

🏗️ Tech Stack
Component	Technology
Frontend	HTML5, JavaScript, Tailwind CSS, Chart.js
Backend	Python (Flask, Flask-CORS)
Data Source	Yahoo Finance via yfinance
Prediction Model	Simple moving average / trend extrapolation
Storage	Browser localStorage (for watchlist)🧩 How It Works

The user enters a stock ticker in the search box.

The frontend sends a request to the Flask API hosted locally at http://127.0.0.1:5000.

The backend retrieves:

Stock Details via /stock_details?ticker=...

Historical & Forecast Data via /predict?ticker=...

The backend processes the data and returns it in JSON format.

The frontend then plots interactive charts using Chart.js and displays insights in real time


🧪 Steps to Run the Project
1. Clone the Repository
git clone https://github.com/<your-username>/Indian-Stock-Predictor.git
cd Indian-Stock-Predictor/backend

2. Install Dependencies

Make sure you have Python 3.8+ installed, then run:

pip install -r requirements.txt


If you don’t have the file yet, install manually:

pip install flask flask-cors yfinance pandas

3. Run the Flask Backend
python app.py


It will start running on:

http://127.0.0.1:5000

4. Open the Frontend

Go to the frontend folder and open index.html in your browser.
Make sure the backend is running before using the site.

🧮 Example API Output

For /predict?ticker=RELIANCE.NS:

{
  "historical": [
    {"Date": "2025-10-20", "Close": 2440.75},
    {"Date": "2025-10-21", "Close": 2455.10}
  ],
  "forecast": [
    {"Date": "2025-11-01", "Close": 2475.32},
    {"Date": "2025-11-02", "Close": 2490.25}
  ]
}

💡 Key Features

✔ Real-time stock details and price updates
✔ 7-day forecast visualization
✔ Add stocks to your watchlist
✔ Compare multiple stocks
✔ Toggle currency between INR and USD
✔ Download chart or export data to CSV
✔ Read the latest news and basic sentiment analysis
✔ Clean and responsive Tailwind UI
