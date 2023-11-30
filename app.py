from flask import Flask, request, jsonify
from pymongo import MongoClient
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime as dt
import spacy
from langdetect import detect
from configparser import ConfigParser
from binance.client import Client
import os
import dateutil.parser
from pycoingecko import CoinGeckoAPI

# Initialize spaCy
nlp = spacy.load("en_core_web_sm")

# Initialize VADER sentiment analysis
sid = SentimentIntensityAnalyzer()

app = Flask(__name__)

# Load configuration file
config = ConfigParser()
config.read('config.ini')

# Setup MongoDB Client
mongo_client = MongoClient(config.get('API_Sentiments', 'MongoClient'))
db = mongo_client['Cluster0']
details_collection = db['Sentiment_DetailsTest']
averages_collection = db['Sentiment_AveragesTest']

# Binance API Key and Secret
api_key = os.environ.get(config.get('API_Sentiments', 'APIKey'))
api_secret = os.environ.get(config.get('API_Sentiments', 'APISecret'))
binance_client = Client(api_key, api_secret)

# Function to get the price of Bitcoin or Ethereum at a specific datetime
def get_datetime_binance_price(crypto, datetime_str):
    # Map the input to the appropriate symbol
    symbol_map = {
        "Bitcoin": "BTCUSDT",
        "Ethereum": "ETHUSDT"
    }
    symbol = symbol_map.get(crypto)

    # Check if the symbol is valid
    if not symbol:
        raise ValueError(f"Invalid cryptocurrency: {crypto}")

    specific_time = dt.fromisoformat(datetime_str)
    specific_time_ms = int(specific_time.timestamp() * 1000)  # Convert to milliseconds

    # Fetch the kline that includes the specified datetime
    klines = binance_client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1MINUTE, specific_time_ms, specific_time_ms+60000)

    if klines:
        close_price = klines[0][4]
        return close_price
    else:
        return "No data available for the specified time"
    
def get_current_binance_price(crypto):
    symbol_map = {
        "Bitcoin": "BTCUSDT",
        "Ethereum": "ETHUSDT"
    }
    symbol = symbol_map.get(crypto)

    if not symbol:
        raise ValueError(f"Invalid cryptocurrency: {crypto}")

    ticker = binance_client.get_symbol_ticker(symbol=symbol)
    return ticker["price"]

def get_coingecko_price(crypto):
    cg = CoinGeckoAPI()
    coin_map = {
        "Bitcoin": "bitcoin",
        "Ethereum": "ethereum"
    }
    coin_id = coin_map.get(crypto)

    if not coin_id:
        raise ValueError(f"Invalid cryptocurrency: {crypto}")

    try:
        price = cg.get_price(ids=coin_id, vs_currencies='usd')
        return price[coin_id]['usd']
    except Exception as e:
        return f"Error fetching data from CoinGecko: {e}"

def extract_financial_mention(text):
    doc = nlp(text)
    financial_terms = [
    'buy', 'sell', 'long', 'short', 'bull', 'bear', 'bullish', 'bearish',
    'price', 'prices', 'cost', 'costs', 'high', 'low', 'peak', 'bottom',
    'value', 'values', 'rate', 'rates', 'valuation', 'valuations',
    'USD', '$', 'dollar', 'dollars', 'euro', 'yen', 'currency', 'currencies',
    'invest', 'investment', 'investor', 'investors', 'divest', 'portfolio',
    'market', 'markets', 'stock', 'stocks', 'share', 'shares',
    'economy', 'economic', 'economist', 'GDP', 'inflation', 'deflation',
    'trade', 'trading', 'trader', 'traders', 'trading volume',
    'financial', 'finance', 'financials', 'capital', 'gain', 'loss',
    'profit', 'margin', 'revenue', 'bear market', 'bull market',
    'IPO', 'initial public offering', 'equity', 'debt',
    'futures', 'options', 'derivatives', 'commodities', 'bonds', 'treasury',
    'hedge', 'hedge fund', 'leverage', 'liquid', 'liquidity',
    'volatility', 'volatile', 'index', 'indices', 'benchmark',
    'speculate', 'speculation', 'risk', 'assessment', 'analysis', 'analyst',
    'dividend', 'yield', 'return', 'ROI', 'return on investment',
    'credit', 'debt', 'loan', 'interest', 'rate cut', 'hike', 'policy',
    'regulation', 'regulatory', 'compliance', 'audit',
    'merge', 'merger', 'acquisition', 'takeover',
    'fiscal', 'monetary', 'quantitative easing', 'stimulus',
    'recession', 'depression', 'recovery', 'boom', 'bust',
    'bubble', 'correction', 'rally', 'crash', 'slump', 'growth',
    'capital gains', 'capital loss', 'asset', 'liability',
    'equities', 'fixed income', 'exchange', 'broker', 'brokerage',
    'arbitrage', 'spread', 'bid', 'ask', 'quote', 'position',
    'sector', 'industry', 'diversify', 'diversification',
    'blue chip', 'penny stocks', 'due diligence',
    'fintech', 'cryptocurrency', 'crypto', 'blockchain', 'Bitcoin', 'Ethereum',
    'Litecoin', 'Ripple', 'wallet', 'token', 'coin', 'mining', 'hashrate',
    'ICO', 'initial coin offering', 'decentralized', 'smart contract',
    'exchange-traded fund', 'ETF', 'mutual fund', 'hedge',
    'technical analysis', 'fundamental analysis', 'chart', 'trend',
    'moving average', 'support', 'resistance', 'indicator',
    'oscillator', 'volume', 'momentum', 'RSI', 'MACD'
]

    financial_entities = [
    "MONEY", "ORG", "GPE", "CARDINAL", "PERCENT", "PRODUCT", "EVENT", 
    "WORK_OF_ART", "LAW", "DATE", "TIME", "QUANTITY", "ORDINAL"
]

    return any(ent.label_ in financial_entities for ent in doc.ents) or \
           any(financial_term in text.lower() for financial_term in financial_terms)

@app.route('/store-text', methods=['POST'])
def store_text():
    data = request.json
    if not data or 'entries' not in data:
        return jsonify({'error': 'No data provided.'}), 400

    source = data.get('source', 'Unknown')
    keyword = data.get('keyword', 'Undefined')

    ids = []
    related_entries = []
    total_textblob_sentiment = 0
    total_vader_sentiment = 0
    total_crypto_price = 0
    count_textblob_entries = 0
    count_vader_entries = 0
    count_crypto_prices = 0

    for entry in data['entries']:
        combined_text = entry.get('title', '') + ' ' + entry['text']

        try:
            lang = detect(combined_text)
        except:
            lang = 'not English'

        if lang != 'en':
            continue

        if extract_financial_mention(combined_text):
            textblob_sentiment = TextBlob(combined_text).sentiment.polarity
            vader_sentiment = sid.polarity_scores(combined_text)['compound']

            if textblob_sentiment != 0 or vader_sentiment != 0:

                entry_date = dateutil.parser.parse(entry['date'])
                crypto_price = get_datetime_binance_price(keyword, entry['date']) 
                crypto_price = float(crypto_price) if crypto_price is not None else 0
                
                if crypto_price == 0:
                    crypto_price = get_current_binance_price(keyword)
                    crypto_price = float(crypto_price) if crypto_price is not None else 0

                if crypto_price == 0:
                    crypto_price = get_coingecko_price(keyword)
                    crypto_price = float(crypto_price) if crypto_price is not None else 0

                if crypto_price != 0:
				
                    entry_to_store = {
                        'user': entry['user'],
                        'title': entry.get('title', ''),
                        'text': entry['text'],
                        'date': entry_date,
                        'textblob_sentiment': textblob_sentiment,
                        'vader_sentiment': vader_sentiment,
                        'price': crypto_price
                    }
                    result = details_collection.insert_one(entry_to_store)
                    ids.append(str(result.inserted_id))
                    related_entries.append(result.inserted_id)

                    if textblob_sentiment != 0:
                        total_textblob_sentiment += textblob_sentiment
                        count_textblob_entries += 1
                    if vader_sentiment != 0:
                        total_vader_sentiment += vader_sentiment
                        count_vader_entries += 1
                    if crypto_price != 0:
                        total_crypto_price += crypto_price
                        count_crypto_prices += 1

    average_textblob_sentiment = total_textblob_sentiment / count_textblob_entries if count_textblob_entries > 0 else 0
    average_vader_sentiment = total_vader_sentiment / count_vader_entries if count_vader_entries > 0 else 0
    average_crypto_price = total_crypto_price / count_crypto_prices if count_crypto_prices > 0 else 0

    if average_textblob_sentiment != 0 and average_vader_sentiment != 0:
        average_entry = {
            'source': source,
            'keyword': keyword,
            'timestamp': dt.now(),
            'textblob_sentiment': average_textblob_sentiment,
            'vader_sentiment': average_vader_sentiment,
            'price': average_crypto_price
        }
        average_result = averages_collection.insert_one(average_entry)
        average_id = average_result.inserted_id

        for entry_id in related_entries:
            details_collection.update_one({'_id': entry_id}, {'$set': {'average_id': str(average_id)}})

        return jsonify({'ids': ids, 'average_id': str(average_id), 'msg': 'Texts and average sentiment stored'}), 200
    else:
        return jsonify({'ids': ids, 'msg': 'Texts stored, no average sentiment stored due to null or zero values'}), 200

if __name__ == '__main__':
    app.run(debug=True)
