import logging
import json
import threading
from kafka import KafkaConsumer, KafkaProducer
from typing import Dict, List, Any, Optional, Callable
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MCPKafkaConnector")

class KafkaConfig(BaseModel):
    """Kafka connection configuration."""
    bootstrap_servers: str = Field(default="localhost:9092", 
                                  description="Comma-separated list of Kafka bootstrap servers")
    consumer_group_id: str = Field(default="crew_chain_consumer", 
                                  description="Consumer group ID for Kafka consumers")
    auto_offset_reset: str = Field(default="latest", 
                                  description="Auto offset reset policy: 'earliest' or 'latest'")
    enable_auto_commit: bool = Field(default=True, 
                                    description="Whether to enable auto-commit for consumers")
    security_protocol: Optional[str] = Field(default=None, 
                                           description="Security protocol: None, 'SSL', 'SASL_PLAINTEXT', or 'SASL_SSL'")
    sasl_mechanism: Optional[str] = Field(default=None, 
                                        description="SASL mechanism: None, 'PLAIN', 'GSSAPI', 'SCRAM-SHA-256', 'SCRAM-SHA-512'")
    sasl_username: Optional[str] = Field(default=None, 
                                       description="SASL username for authentication")
    sasl_password: Optional[str] = Field(default=None, 
                                       description="SASL password for authentication")

class MessageSchema(BaseModel):
    """Base schema for MCP messages."""
    message_id: str = Field(description="Unique identifier for the message")
    timestamp: str = Field(description="ISO format timestamp when the message was created")
    source: str = Field(description="Source system that generated the message")
    message_type: str = Field(description="Type of message")
    payload: Dict[str, Any] = Field(description="Message payload data")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional metadata")

class MCPKafkaConnector:
    """Connector for Message Consumption Protocol (MCP) via Kafka."""
    
    def __init__(self, config: KafkaConfig):
        """
        Initialize the MCP Kafka connector.
        
        Args:
            config: Kafka connection configuration
        """
        self.config = config
        self.producer = None
        self.consumers = {}
        self.consumer_threads = {}
        self.is_running = False
        self.lock = threading.Lock()
        
        self._initialize_producer()
    
    def _initialize_producer(self):
        """Initialize the Kafka producer."""
        try:
            # Set up security config if provided
            config_kwargs = {}
            if self.config.security_protocol:
                config_kwargs['security_protocol'] = self.config.security_protocol
                
                if self.config.sasl_mechanism:
                    config_kwargs['sasl_mechanism'] = self.config.sasl_mechanism
                    
                if self.config.sasl_username and self.config.sasl_password:
                    config_kwargs['sasl_plain_username'] = self.config.sasl_username
                    config_kwargs['sasl_plain_password'] = self.config.sasl_password
            
            self.producer = KafkaProducer(
                bootstrap_servers=self.config.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                **config_kwargs
            )
            logger.info("Kafka producer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {str(e)}")
            raise
    
    def create_consumer(self, topic: str, message_handler: Callable[[Dict], None], 
                      auto_start: bool = True) -> str:
        """
        Create a Kafka consumer for a specific topic.
        
        Args:
            topic: Kafka topic to consume from
            message_handler: Callback function to handle received messages
            auto_start: Whether to automatically start the consumer
            
        Returns:
            Consumer ID that can be used to reference this consumer
        """
        with self.lock:
            consumer_id = f"{topic}_{datetime.now().timestamp()}"
            
            # Set up security config if provided
            config_kwargs = {}
            if self.config.security_protocol:
                config_kwargs['security_protocol'] = self.config.security_protocol
                
                if self.config.sasl_mechanism:
                    config_kwargs['sasl_mechanism'] = self.config.sasl_mechanism
                    
                if self.config.sasl_username and self.config.sasl_password:
                    config_kwargs['sasl_plain_username'] = self.config.sasl_username
                    config_kwargs['sasl_plain_password'] = self.config.sasl_password
            
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=self.config.bootstrap_servers,
                group_id=self.config.consumer_group_id,
                auto_offset_reset=self.config.auto_offset_reset,
                enable_auto_commit=self.config.enable_auto_commit,
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                **config_kwargs
            )
            
            self.consumers[consumer_id] = {
                'consumer': consumer,
                'topic': topic,
                'handler': message_handler,
                'active': False
            }
            
            logger.info(f"Created consumer {consumer_id} for topic {topic}")
            
            if auto_start:
                self.start_consumer(consumer_id)
                
            return consumer_id
    
    def start_consumer(self, consumer_id: str) -> bool:
        """
        Start a consumer by its ID.
        
        Args:
            consumer_id: ID of the consumer to start
            
        Returns:
            True if the consumer was started, False otherwise
        """
        with self.lock:
            if consumer_id not in self.consumers:
                logger.warning(f"Consumer {consumer_id} not found")
                return False
                
            if self.consumers[consumer_id]['active']:
                logger.warning(f"Consumer {consumer_id} is already active")
                return True
                
            consumer_info = self.consumers[consumer_id]
            consumer_thread = threading.Thread(
                target=self._consumer_loop,
                args=(consumer_id, consumer_info['consumer'], consumer_info['handler']),
                daemon=True
            )
            consumer_thread.start()
            
            self.consumer_threads[consumer_id] = consumer_thread
            self.consumers[consumer_id]['active'] = True
            
            logger.info(f"Started consumer {consumer_id} for topic {consumer_info['topic']}")
            return True
    
    def stop_consumer(self, consumer_id: str) -> bool:
        """
        Stop a consumer by its ID.
        
        Args:
            consumer_id: ID of the consumer to stop
            
        Returns:
            True if the consumer was stopped, False otherwise
        """
        with self.lock:
            if consumer_id not in self.consumers:
                logger.warning(f"Consumer {consumer_id} not found")
                return False
                
            if not self.consumers[consumer_id]['active']:
                logger.warning(f"Consumer {consumer_id} is not active")
                return True
                
            # Mark consumer as inactive to signal thread to stop
            self.consumers[consumer_id]['active'] = False
            
            logger.info(f"Stopping consumer {consumer_id}")
            return True
    
    def _consumer_loop(self, consumer_id: str, consumer: KafkaConsumer, handler: Callable[[Dict], None]):
        """
        Main loop for consuming messages from Kafka.
        
        Args:
            consumer_id: ID of the consumer
            consumer: Kafka consumer instance
            handler: Callback function to handle received messages
        """
        logger.info(f"Consumer {consumer_id} loop started")
        
        try:
            for message in consumer:
                # Check if consumer is still active
                with self.lock:
                    if consumer_id not in self.consumers or not self.consumers[consumer_id]['active']:
                        break
                
                try:
                    # Handle the message
                    value = message.value
                    logger.debug(f"Consumer {consumer_id} received message: {value}")
                    
                    # Call the handler with the message
                    handler(value)
                    
                except Exception as e:
                    logger.error(f"Error handling message in consumer {consumer_id}: {str(e)}")
        except Exception as e:
            logger.error(f"Error in consumer {consumer_id} loop: {str(e)}")
        finally:
            logger.info(f"Consumer {consumer_id} loop ended")
            
            try:
                consumer.close()
                logger.info(f"Consumer {consumer_id} closed")
            except Exception as e:
                logger.error(f"Error closing consumer {consumer_id}: {str(e)}")
                
            # Clean up consumer references
            with self.lock:
                if consumer_id in self.consumers:
                    self.consumers[consumer_id]['active'] = False
                    
                if consumer_id in self.consumer_threads:
                    del self.consumer_threads[consumer_id]
    
    def send_message(self, topic: str, message: Dict[str, Any]) -> bool:
        """
        Send a message to a Kafka topic.
        
        Args:
            topic: Kafka topic to send the message to
            message: Message to send
            
        Returns:
            True if the message was sent successfully, False otherwise
        """
        if not self.producer:
            logger.error("Producer not initialized")
            return False
            
        try:
            # Add timestamp if not present
            if 'timestamp' not in message:
                message['timestamp'] = datetime.now().isoformat()
                
            # Send the message
            future = self.producer.send(topic, message)
            
            # Block until the message is sent
            future.get(timeout=10)
            
            logger.debug(f"Message sent to topic {topic}: {message}")
            return True
        except Exception as e:
            logger.error(f"Error sending message to topic {topic}: {str(e)}")
            return False
    
    def close(self):
        """Close all consumers and the producer."""
        logger.info("Closing MCP Kafka connector")
        
        # Stop all consumers
        with self.lock:
            for consumer_id in list(self.consumers.keys()):
                self.stop_consumer(consumer_id)
        
        # Close the producer
        if self.producer:
            try:
                self.producer.close()
                logger.info("Producer closed")
            except Exception as e:
                logger.error(f"Error closing producer: {str(e)}")


class MCPKafkaInput(BaseModel):
    """Input schema for MCP Kafka tools."""
    topic: str = Field(..., description="Kafka topic to interact with")
    message_type: str = Field(..., description="Type of message to send: 'market_data', 'trade_signal', 'order', 'validation'")
    payload: Dict[str, Any] = Field(..., description="Message payload to send")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional metadata for the message")


class MCPKafkaProducerTool(BaseTool):
    """Tool for sending messages to Kafka topics using MCP."""
    name = "mcp_kafka_producer"
    description = "Send cryptocurrency trading data and signals to the MCP Kafka system."
    args_schema = MCPKafkaInput
    
    def __init__(self, config: Optional[KafkaConfig] = None):
        """
        Initialize the MCP Kafka producer tool.
        
        Args:
            config: Kafka configuration or None to use defaults
        """
        super().__init__()
        self.config = config or KafkaConfig()
        self.connector = MCPKafkaConnector(self.config)
        
    def _run(self, topic: str, message_type: str, payload: Dict[str, Any], 
           metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Send a message to a Kafka topic using MCP.
        
        Args:
            topic: Kafka topic to send the message to
            message_type: Type of message
            payload: Message payload
            metadata: Additional metadata
            
        Returns:
            Success or error message
        """
        try:
            # Create MCP message structure
            message = {
                "message_id": f"{message_type}_{datetime.now().timestamp()}",
                "timestamp": datetime.now().isoformat(),
                "source": "crew_chain",
                "message_type": message_type,
                "payload": payload,
                "metadata": metadata or {}
            }
            
            # Send the message
            success = self.connector.send_message(topic, message)
            
            if success:
                return f"Successfully sent {message_type} message to {topic}"
            else:
                return f"Failed to send {message_type} message to {topic}"
        except Exception as e:
            return f"Error sending message to Kafka: {str(e)}"
    
    def cleanup(self):
        """Clean up resources when the tool is no longer needed."""
        if hasattr(self, 'connector'):
            self.connector.close()


class MCPKafkaConsumerTool(BaseTool):
    """Tool for consuming messages from Kafka topics using MCP."""
    name = "mcp_kafka_consumer"
    description = "Register a handler for consuming cryptocurrency trading data and signals from the MCP Kafka system."
    
    def __init__(self, config: Optional[KafkaConfig] = None):
        """
        Initialize the MCP Kafka consumer tool.
        
        Args:
            config: Kafka configuration or None to use defaults
        """
        super().__init__()
        self.config = config or KafkaConfig()
        self.connector = MCPKafkaConnector(self.config)
        self.registered_handlers = {}
        
    def _run(self, topic: str, handler_name: str) -> str:
        """
        Register a handler for a Kafka topic using MCP.
        
        Args:
            topic: Kafka topic to consume from
            handler_name: Name of the handler function to register
            
        Returns:
            Success or error message
        """
        try:
            # Check if handler exists
            if handler_name not in self.registered_handlers:
                return f"Handler {handler_name} not found. Please register a handler first."
                
            handler = self.registered_handlers[handler_name]
            
            # Create consumer
            consumer_id = self.connector.create_consumer(topic, handler)
            
            return f"Successfully registered handler {handler_name} for topic {topic} (Consumer ID: {consumer_id})"
        except Exception as e:
            return f"Error registering Kafka consumer: {str(e)}"
    
    def register_handler(self, name: str, handler_func: Callable[[Dict], None]):
        """
        Register a message handler function.
        
        Args:
            name: Name to identify the handler
            handler_func: Function to call when a message is received
        """
        self.registered_handlers[name] = handler_func
        logger.info(f"Registered message handler: {name}")
    
    def cleanup(self):
        """Clean up resources when the tool is no longer needed."""
        if hasattr(self, 'connector'):
            self.connector.close()


# Example of a message handler function
def example_price_data_handler(message: Dict[str, Any]):
    """
    Example handler for cryptocurrency price data.
    
    Args:
        message: Message received from Kafka
    """
    try:
        # Extract data from the message
        timestamp = message.get("timestamp", "")
        message_type = message.get("message_type", "")
        payload = message.get("payload", {})
        
        if message_type == "market_data":
            # Process market data
            symbol = payload.get("symbol", "")
            price = payload.get("price", 0.0)
            volume = payload.get("volume", 0.0)
            
            logger.info(f"Received price data for {symbol}: ${price:.2f} (Volume: {volume:.2f})")
            
            # Here you would typically:
            # 1. Store the data in a database
            # 2. Update any real-time indicators or charts
            # 3. Trigger analysis processes
        
        elif message_type == "trade_signal":
            # Process trade signal
            symbol = payload.get("symbol", "")
            action = payload.get("action", "")
            confidence = payload.get("confidence", 0.0)
            
            logger.info(f"Received trade signal for {symbol}: {action} (Confidence: {confidence:.2f})")
            
            # Here you would typically:
            # 1. Validate the signal
            # 2. Check if it meets your criteria for execution
            # 3. Potentially execute a trade
    
    except Exception as e:
        logger.error(f"Error handling market data message: {str(e)}") 