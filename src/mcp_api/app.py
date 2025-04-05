import os
import json
import logging
import threading
import time
from datetime import datetime
from typing import Dict, Any

from flask import Flask, request, jsonify
from kafka import KafkaProducer, KafkaConsumer
import pymongo
import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MCP-API")

# Initialize Flask app
app = Flask(__name__)

# Get configuration from environment variables
KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://admin:password@localhost:27017/")
POSTGRES_DSN = os.environ.get("POSTGRES_DSN", "postgresql://admin:password@localhost:5432/crypto_trading")

# Initialize Kafka producer
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Initialize MongoDB client
mongo_client = pymongo.MongoClient(MONGODB_URI)
mongo_db = mongo_client.get_database("crypto_trading")
market_data_collection = mongo_db.get_collection("market_data")
trading_signals_collection = mongo_db.get_collection("trading_signals")

# PostgreSQL connection function
def get_postgres_connection():
    return psycopg2.connect(POSTGRES_DSN)

# Ensure PostgreSQL tables exist
def initialize_postgres():
    with get_postgres_connection() as conn:
        with conn.cursor() as cursor:
            # Create market_data table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                price NUMERIC(20, 8) NOT NULL,
                volume NUMERIC(20, 8) NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                source VARCHAR(100) NOT NULL
            );
            """)
            
            # Create trading_signals table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS trading_signals (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                action VARCHAR(20) NOT NULL,
                position_type VARCHAR(20) NOT NULL,
                entry_price NUMERIC(20, 8),
                target_price NUMERIC(20, 8),
                stop_loss NUMERIC(20, 8),
                confidence NUMERIC(5, 2),
                timestamp TIMESTAMP NOT NULL,
                source VARCHAR(100) NOT NULL
            );
            """)
        conn.commit()

# Initialize Kafka consumer threads
def start_kafka_consumers():
    # Market data consumer
    market_data_thread = threading.Thread(
        target=consume_market_data,
        daemon=True
    )
    market_data_thread.start()
    
    # Trading signals consumer
    trading_signals_thread = threading.Thread(
        target=consume_trading_signals,
        daemon=True
    )
    trading_signals_thread.start()

# Kafka consumer for market data
def consume_market_data():
    consumer = KafkaConsumer(
        'market_data',
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id='mcp_api_market_data',
        auto_offset_reset='latest',
        value_deserializer=lambda v: json.loads(v.decode('utf-8'))
    )
    
    logger.info("Starting market data consumer")
    
    for message in consumer:
        try:
            data = message.value
            logger.debug(f"Received market data: {data}")
            
            # Process only if it's a market_data message type
            if data.get('message_type') == 'market_data':
                payload = data.get('payload', {})
                
                # Insert into MongoDB
                market_data_collection.insert_one({
                    'symbol': payload.get('symbol'),
                    'price': payload.get('price'),
                    'volume': payload.get('volume'),
                    'timestamp': datetime.fromisoformat(data.get('timestamp')),
                    'source': data.get('source')
                })
                
                # Insert into PostgreSQL
                with get_postgres_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute("""
                        INSERT INTO market_data (symbol, price, volume, timestamp, source)
                        VALUES (%s, %s, %s, %s, %s)
                        """, (
                            payload.get('symbol'),
                            payload.get('price'),
                            payload.get('volume'),
                            datetime.fromisoformat(data.get('timestamp')),
                            data.get('source')
                        ))
                    conn.commit()
        except Exception as e:
            logger.error(f"Error processing market data: {str(e)}")

# Kafka consumer for trading signals
def consume_trading_signals():
    consumer = KafkaConsumer(
        'trading_signals',
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id='mcp_api_trading_signals',
        auto_offset_reset='latest',
        value_deserializer=lambda v: json.loads(v.decode('utf-8'))
    )
    
    logger.info("Starting trading signals consumer")
    
    for message in consumer:
        try:
            data = message.value
            logger.debug(f"Received trading signal: {data}")
            
            # Process only if it's a trade_signal message type
            if data.get('message_type') == 'trade_signal':
                payload = data.get('payload', {})
                
                # Insert into MongoDB
                trading_signals_collection.insert_one({
                    'symbol': payload.get('symbol'),
                    'action': payload.get('action'),
                    'position_type': payload.get('position_type'),
                    'entry_price': payload.get('entry_price'),
                    'target_price': payload.get('target_price'),
                    'stop_loss': payload.get('stop_loss'),
                    'confidence': payload.get('confidence'),
                    'timestamp': datetime.fromisoformat(data.get('timestamp')),
                    'source': data.get('source')
                })
                
                # Insert into PostgreSQL
                with get_postgres_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute("""
                        INSERT INTO trading_signals 
                        (symbol, action, position_type, entry_price, target_price, stop_loss, confidence, timestamp, source)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            payload.get('symbol'),
                            payload.get('action'),
                            payload.get('position_type'),
                            payload.get('entry_price'),
                            payload.get('target_price'),
                            payload.get('stop_loss'),
                            payload.get('confidence'),
                            datetime.fromisoformat(data.get('timestamp')),
                            data.get('source')
                        ))
                    conn.commit()
        except Exception as e:
            logger.error(f"Error processing trading signal: {str(e)}")

# Route for health check
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat()
    })

# Route for sending market data
@app.route('/market-data', methods=['POST'])
def send_market_data():
    try:
        data = request.json
        
        # Validate required fields
        if not data.get('symbol') or not isinstance(data.get('price'), (int, float)):
            return jsonify({
                'status': 'error',
                'message': 'Missing required fields: symbol and price are required'
            }), 400
            
        # Create MCP message structure
        message = {
            'message_id': f"market_data_{datetime.now().timestamp()}",
            'timestamp': datetime.now().isoformat(),
            'source': 'mcp_api',
            'message_type': 'market_data',
            'payload': {
                'symbol': data.get('symbol').upper(),
                'price': float(data.get('price')),
                'volume': float(data.get('volume', 0)),
                'high_24h': data.get('high_24h'),
                'low_24h': data.get('low_24h'),
                'change_24h': data.get('change_24h')
            },
            'metadata': data.get('metadata', {})
        }
        
        # Send to Kafka
        future = producer.send('market_data', message)
        result = future.get(timeout=10)
        
        return jsonify({
            'status': 'success',
            'message': 'Market data sent successfully',
            'topic': 'market_data',
            'partition': result.partition,
            'offset': result.offset
        })
    except Exception as e:
        logger.error(f"Error sending market data: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Error sending market data: {str(e)}"
        }), 500

# Route for sending trading signals
@app.route('/trading-signal', methods=['POST'])
def send_trading_signal():
    try:
        data = request.json
        
        # Validate required fields
        if not data.get('symbol') or not data.get('action'):
            return jsonify({
                'status': 'error',
                'message': 'Missing required fields: symbol and action are required'
            }), 400
            
        # Create MCP message structure
        message = {
            'message_id': f"trade_signal_{datetime.now().timestamp()}",
            'timestamp': datetime.now().isoformat(),
            'source': 'mcp_api',
            'message_type': 'trade_signal',
            'payload': {
                'symbol': data.get('symbol').upper(),
                'action': data.get('action').lower(),
                'position_type': data.get('position_type', 'long').lower(),
                'entry_price': data.get('entry_price'),
                'target_price': data.get('target_price'),
                'stop_loss': data.get('stop_loss'),
                'confidence': data.get('confidence', 50.0)
            },
            'metadata': data.get('metadata', {})
        }
        
        # Send to Kafka
        future = producer.send('trading_signals', message)
        result = future.get(timeout=10)
        
        return jsonify({
            'status': 'success',
            'message': 'Trading signal sent successfully',
            'topic': 'trading_signals',
            'partition': result.partition,
            'offset': result.offset
        })
    except Exception as e:
        logger.error(f"Error sending trading signal: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Error sending trading signal: {str(e)}"
        }), 500

# Route for getting recent market data
@app.route('/market-data/<symbol>', methods=['GET'])
def get_market_data(symbol):
    try:
        symbol = symbol.upper()
        limit = int(request.args.get('limit', 10))
        
        # Get data from PostgreSQL
        with get_postgres_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                SELECT * FROM market_data
                WHERE symbol = %s
                ORDER BY timestamp DESC
                LIMIT %s
                """, (symbol, limit))
                
                results = cursor.fetchall()
                
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'count': len(results),
            'data': results
        })
    except Exception as e:
        logger.error(f"Error getting market data: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Error getting market data: {str(e)}"
        }), 500

# Route for getting recent trading signals
@app.route('/trading-signals/<symbol>', methods=['GET'])
def get_trading_signals(symbol):
    try:
        symbol = symbol.upper()
        limit = int(request.args.get('limit', 10))
        
        # Get data from PostgreSQL
        with get_postgres_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                SELECT * FROM trading_signals
                WHERE symbol = %s
                ORDER BY timestamp DESC
                LIMIT %s
                """, (symbol, limit))
                
                results = cursor.fetchall()
                
        return jsonify({
            'status': 'success',
            'symbol': symbol,
            'count': len(results),
            'data': results
        })
    except Exception as e:
        logger.error(f"Error getting trading signals: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Error getting trading signals: {str(e)}"
        }), 500

# Initialize everything on startup
@app.before_first_request
def initialize():
    try:
        # Initialize PostgreSQL tables
        initialize_postgres()
        
        # Start Kafka consumers
        start_kafka_consumers()
        
        logger.info("MCP API initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing MCP API: {str(e)}")

# Main entry point
if __name__ == "__main__":
    # Initialize PostgreSQL tables
    initialize_postgres()
    
    # Start Kafka consumers
    start_kafka_consumers()
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000) 