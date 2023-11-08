from flask import Flask, request, jsonify
from pymongo import MongoClient
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime
import spacy
import requests
from langdetect import detect

# Initialize spaCy
nlp = spacy.load("en_core_web_sm")

# Initialize VADER sentiment analysis
sid = SentimentIntensityAnalyzer()

app = Flask(__name__)

# Setup MongoDB Client
client = MongoClient("mongodb://localhost:27017/")
db = client['SocialMedia_Analysis']
test_collection = db['Sentiment_Details']
average_collection = db['Sentiment_Average']

def get_bitcoin_price():
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data["bitcoin"]["usd"]
    except requests.RequestException as e:
        print(f"Error fetching Bitcoin price: {e}")
        return None

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
    count_textblob_entries = 0
    count_vader_entries = 0

    for entry in data['entries']:
        # Detect the language of the text
        try:
            lang = detect(entry['text'])
        except:
            # If language detection fails, assume the language is not English
            lang = 'not English'

        # If the text is not English, skip to the next entry
        if lang != 'en':
            continue

        textblob_sentiment = TextBlob(entry['text']).sentiment.polarity
        vader_sentiment = sid.polarity_scores(entry['text'])['compound']

        # Skip this entry if both sentiment scores are 0
        if textblob_sentiment == 0 and vader_sentiment == 0:
            continue

        if extract_financial_mention(entry['text']):
            if textblob_sentiment != 0:
                total_textblob_sentiment += textblob_sentiment
                count_textblob_entries += 1
            if vader_sentiment != 0:
                total_vader_sentiment += vader_sentiment
                count_vader_entries += 1

            entry_to_store = {
                'user': entry['user'],
                'text': entry['text'],
                'date': datetime.fromisoformat(entry['date']),
                'textblob_sentiment': textblob_sentiment,
                'vader_sentiment': vader_sentiment
            }
            result = test_collection.insert_one(entry_to_store)
            ids.append(str(result.inserted_id))
            related_entries.append(result.inserted_id)

    # Calculate averages independently for TextBlob and VADER, only if their counts are greater than zero
    average_textblob_sentiment = (total_textblob_sentiment / count_textblob_entries) if count_textblob_entries > 0 else None
    average_vader_sentiment = (total_vader_sentiment / count_vader_entries) if count_vader_entries > 0 else None

    average_entry = {
        'source': source,
        'keyword': keyword,
        'timestamp': datetime.now(),
        'textblob_sentiment': average_textblob_sentiment,
        'vader_sentiment': average_vader_sentiment,
        'price': get_bitcoin_price() if keyword.lower() == 'bitcoin' else None
    }

    average_result = average_collection.insert_one(average_entry)
    average_id = average_result.inserted_id

    for entry_id in related_entries:
        test_collection.update_one({'_id': entry_id}, {'$set': {'average_id': average_id}})

    return jsonify({'ids': ids, 'msg': 'Texts stored', 'average_id': str(average_id)}), 200

if __name__ == '__main__':
    app.run(debug=True)