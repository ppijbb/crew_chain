import schedule
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Any, Optional
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RecurringOperations")

class ScheduledTask:
    """Class representing a scheduled task with execution state tracking."""
    
    def __init__(self, task_id: str, task_name: str, callback: Callable, 
                 args: List[Any] = None, kwargs: Dict[str, Any] = None):
        """
        Initialize a scheduled task.
        
        Args:
            task_id: Unique identifier for the task
            task_name: Human-readable name for the task
            callback: Function to call when the task is executed
            args: Positional arguments to pass to the callback
            kwargs: Keyword arguments to pass to the callback
        """
        self.task_id = task_id
        self.task_name = task_name
        self.callback = callback
        self.args = args or []
        self.kwargs = kwargs or {}
        self.last_execution = None
        self.next_execution = None
        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.is_active = True
        self.created_at = datetime.now()
        self.status = "scheduled"
        self.last_error = None
        
    def execute(self):
        """Execute the task and update execution statistics."""
        if not self.is_active:
            logger.info(f"Task {self.task_id} is inactive and will not be executed.")
            return
            
        self.last_execution = datetime.now()
        self.execution_count += 1
        self.status = "running"
        
        try:
            logger.info(f"Executing task {self.task_id}: {self.task_name}")
            result = self.callback(*self.args, **self.kwargs)
            self.success_count += 1
            self.status = "succeeded"
            logger.info(f"Task {self.task_id} executed successfully")
            return result
        except Exception as e:
            self.failure_count += 1
            self.status = "failed"
            self.last_error = str(e)
            logger.error(f"Task {self.task_id} failed: {str(e)}")
            
    def deactivate(self):
        """Deactivate the task so it will not be executed again."""
        self.is_active = False
        self.status = "deactivated"
        
    def reactivate(self):
        """Reactivate a previously deactivated task."""
        self.is_active = True
        self.status = "scheduled"
        
    def get_status(self) -> dict:
        """Get the current status of the task."""
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "is_active": self.is_active,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "last_execution": self.last_execution.isoformat() if self.last_execution else None,
            "next_execution": self.next_execution.isoformat() if self.next_execution else None,
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "last_error": self.last_error
        }


class RecurringOperationManager:
    """Manager for recurring operations and scheduled tasks."""
    
    def __init__(self):
        """Initialize the recurring operation manager."""
        self.tasks = {}
        self.scheduler_thread = None
        self.is_running = False
        self.lock = threading.Lock()
    
    def start(self):
        """Start the scheduler in a background thread."""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
            
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        logger.info("Recurring operation scheduler started")
        
    def stop(self):
        """Stop the scheduler thread."""
        if not self.is_running:
            logger.warning("Scheduler is not running")
            return
            
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=1.0)
        logger.info("Recurring operation scheduler stopped")
    
    def _scheduler_loop(self):
        """Main loop for the scheduler thread."""
        while self.is_running:
            with self.lock:
                schedule.run_pending()
            time.sleep(1)
    
    def schedule_task(self, task_id: str, task_name: str, callback: Callable,
                    schedule_type: str, interval: int = None, 
                    day_of_week: str = None, time_of_day: str = None,
                    args: List[Any] = None, kwargs: Dict[str, Any] = None) -> str:
        """
        Schedule a new task with the specified parameters.
        
        Args:
            task_id: Unique identifier for the task, or None to generate one
            task_name: Human-readable name for the task
            callback: Function to call when the task is executed
            schedule_type: Type of schedule ('interval', 'daily', 'weekly')
            interval: Interval in minutes (for 'interval' schedule)
            day_of_week: Day of week (for 'weekly' schedule)
            time_of_day: Time of day in HH:MM format (for 'daily' and 'weekly' schedules)
            args: Positional arguments to pass to the callback
            kwargs: Keyword arguments to pass to the callback
            
        Returns:
            The task_id of the scheduled task
        """
        # Generate a task ID if none was provided
        if not task_id:
            task_id = f"task_{int(time.time())}_{hash(task_name) % 10000}"
            
        # Check if task already exists
        if task_id in self.tasks:
            logger.warning(f"Task {task_id} already exists. Overwriting.")
            self.cancel_task(task_id)
        
        # Create the task object
        task = ScheduledTask(task_id, task_name, callback, args, kwargs)
        
        # Schedule based on the schedule type
        with self.lock:
            job = None
            
            if schedule_type == 'interval':
                if not interval or interval < 1:
                    raise ValueError("Interval must be a positive number of minutes")
                    
                def task_wrapper():
                    task.execute()
                    # Update next execution time
                    task.next_execution = datetime.now() + timedelta(minutes=interval)
                    
                job = schedule.every(interval).minutes.do(task_wrapper)
                # Set initial next execution time
                task.next_execution = datetime.now() + timedelta(minutes=interval)
                
            elif schedule_type == 'daily':
                if not time_of_day:
                    raise ValueError("Time of day must be specified for daily schedules")
                    
                try:
                    hour, minute = map(int, time_of_day.split(':'))
                except ValueError:
                    raise ValueError("Time of day must be in HH:MM format")
                    
                def task_wrapper():
                    task.execute()
                    # Update next execution time
                    next_day = datetime.now() + timedelta(days=1)
                    task.next_execution = datetime(next_day.year, next_day.month, next_day.day, 
                                                hour, minute)
                    
                job = schedule.every().day.at(time_of_day).do(task_wrapper)
                # Set initial next execution time
                now = datetime.now()
                task_time = datetime(now.year, now.month, now.day, hour, minute)
                if task_time <= now:
                    task_time += timedelta(days=1)
                task.next_execution = task_time
                
            elif schedule_type == 'weekly':
                if not day_of_week or not time_of_day:
                    raise ValueError("Day of week and time of day must be specified for weekly schedules")
                    
                try:
                    hour, minute = map(int, time_of_day.split(':'))
                except ValueError:
                    raise ValueError("Time of day must be in HH:MM format")
                    
                valid_days = {
                    'monday': schedule.every().monday,
                    'tuesday': schedule.every().tuesday,
                    'wednesday': schedule.every().wednesday,
                    'thursday': schedule.every().thursday,
                    'friday': schedule.every().friday,
                    'saturday': schedule.every().saturday,
                    'sunday': schedule.every().sunday
                }
                
                day_of_week = day_of_week.lower()
                if day_of_week not in valid_days:
                    raise ValueError(f"Invalid day of week: {day_of_week}")
                    
                def task_wrapper():
                    task.execute()
                    # Update next execution time
                    task.next_execution = task.next_execution + timedelta(days=7)
                    
                job = valid_days[day_of_week].at(time_of_day).do(task_wrapper)
                
                # Set initial next execution time
                now = datetime.now()
                days_ahead = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 
                             'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6}
                target_day = days_ahead[day_of_week]
                current_day = now.weekday()
                days_until = (target_day - current_day) % 7
                
                next_execution = now + timedelta(days=days_until)
                next_execution = datetime(next_execution.year, next_execution.month, 
                                        next_execution.day, hour, minute)
                
                if days_until == 0 and now.time() > datetime.strptime(time_of_day, "%H:%M").time():
                    next_execution += timedelta(days=7)
                    
                task.next_execution = next_execution
                
            else:
                raise ValueError(f"Invalid schedule type: {schedule_type}")
                
            # Store the task
            self.tasks[task_id] = {
                'task': task,
                'job': job,
                'schedule_type': schedule_type,
                'schedule_params': {
                    'interval': interval,
                    'day_of_week': day_of_week,
                    'time_of_day': time_of_day
                }
            }
            
            logger.info(f"Scheduled task {task_id}: {task_name} ({schedule_type})")
            return task_id
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a scheduled task.
        
        Args:
            task_id: ID of the task to cancel
            
        Returns:
            True if the task was cancelled, False if it was not found
        """
        with self.lock:
            if task_id in self.tasks:
                task_info = self.tasks[task_id]
                schedule.cancel_job(task_info['job'])
                task_info['task'].deactivate()
                del self.tasks[task_id]
                logger.info(f"Cancelled task {task_id}")
                return True
            else:
                logger.warning(f"Task {task_id} not found")
                return False
    
    def pause_task(self, task_id: str) -> bool:
        """
        Pause a scheduled task without removing it.
        
        Args:
            task_id: ID of the task to pause
            
        Returns:
            True if the task was paused, False if it was not found
        """
        with self.lock:
            if task_id in self.tasks:
                task_info = self.tasks[task_id]
                task_info['task'].deactivate()
                logger.info(f"Paused task {task_id}")
                return True
            else:
                logger.warning(f"Task {task_id} not found")
                return False
    
    def resume_task(self, task_id: str) -> bool:
        """
        Resume a paused task.
        
        Args:
            task_id: ID of the task to resume
            
        Returns:
            True if the task was resumed, False if it was not found
        """
        with self.lock:
            if task_id in self.tasks:
                task_info = self.tasks[task_id]
                task_info['task'].reactivate()
                logger.info(f"Resumed task {task_id}")
                return True
            else:
                logger.warning(f"Task {task_id} not found")
                return False
    
    def get_task_status(self, task_id: str) -> Optional[dict]:
        """
        Get the status of a scheduled task.
        
        Args:
            task_id: ID of the task to get status for
            
        Returns:
            Task status dictionary or None if task not found
        """
        with self.lock:
            if task_id in self.tasks:
                task_info = self.tasks[task_id]
                status = task_info['task'].get_status()
                status.update({
                    'schedule_type': task_info['schedule_type'],
                    'schedule_params': task_info['schedule_params']
                })
                return status
            else:
                logger.warning(f"Task {task_id} not found")
                return None
    
    def list_tasks(self) -> List[dict]:
        """
        List all scheduled tasks.
        
        Returns:
            List of task status dictionaries
        """
        with self.lock:
            result = []
            for task_id, task_info in self.tasks.items():
                status = task_info['task'].get_status()
                status.update({
                    'schedule_type': task_info['schedule_type'],
                    'schedule_params': task_info['schedule_params']
                })
                result.append(status)
            return result


class RecurringOperationInput(BaseModel):
    """Input schema for the recurring operation tool."""
    operation_type: str = Field(..., description="Type of recurring operation: 'market_scan', 'portfolio_rebalance', 'trade_verification', 'sentiment_analysis'")
    schedule_type: str = Field(..., description="Type of schedule: 'interval', 'daily', 'weekly'")
    interval: Optional[int] = Field(None, description="Interval in minutes (for 'interval' schedule)")
    day_of_week: Optional[str] = Field(None, description="Day of week (for 'weekly' schedule): 'monday', 'tuesday', etc.")
    time_of_day: Optional[str] = Field(None, description="Time of day in HH:MM format (for 'daily' and 'weekly' schedules)")
    parameters: Optional[dict] = Field(None, description="Additional parameters for the operation")
    task_id: Optional[str] = Field(None, description="Optional task ID for updating existing tasks")


class RecurringOperationTool(BaseTool):
    """Tool for scheduling recurring operations in the crypto trading system."""
    name = "recurring_operation_scheduler"
    description = "Schedule recurring operations such as market scans, portfolio rebalancing, trade verification, and sentiment analysis."
    args_schema = RecurringOperationInput
    
    def __init__(self):
        """Initialize the recurring operation tool."""
        super().__init__()
        self.manager = RecurringOperationManager()
        self.manager.start()
        
        # Define operation handlers
        self.operation_handlers = {
            'market_scan': self._handle_market_scan,
            'portfolio_rebalance': self._handle_portfolio_rebalance,
            'trade_verification': self._handle_trade_verification,
            'sentiment_analysis': self._handle_sentiment_analysis
        }
        
    def _run(self, operation_type: str, schedule_type: str, interval: Optional[int] = None,
           day_of_week: Optional[str] = None, time_of_day: Optional[str] = None,
           parameters: Optional[dict] = None, task_id: Optional[str] = None) -> str:
        """
        Schedule a recurring operation with the specified parameters.
        """
        try:
            # Validate operation type
            if operation_type not in self.operation_handlers:
                return f"Error: Invalid operation type '{operation_type}'. Valid options are: {', '.join(self.operation_handlers.keys())}"
                
            # Get the handler for this operation type
            handler = self.operation_handlers[operation_type]
            
            # Create a readable task name
            task_name = f"{operation_type.replace('_', ' ').title()}"
            if parameters and 'crypto_symbol' in parameters:
                task_name += f" - {parameters['crypto_symbol'].upper()}"
                
            # Schedule the task
            scheduled_task_id = self.manager.schedule_task(
                task_id=task_id,
                task_name=task_name,
                callback=handler,
                schedule_type=schedule_type,
                interval=interval,
                day_of_week=day_of_week,
                time_of_day=time_of_day,
                kwargs={'parameters': parameters or {}}
            )
            
            # Get the task status for the response
            task_status = self.manager.get_task_status(scheduled_task_id)
            next_execution = task_status['next_execution']
            
            # Return a formatted success message
            return f"""
Successfully scheduled {task_name} operation!

Task ID: {scheduled_task_id}
Schedule Type: {schedule_type.capitalize()}
Next Execution: {next_execution}

Use this Task ID to manage this recurring operation in the future.
"""
        except Exception as e:
            return f"Error scheduling recurring operation: {str(e)}"
    
    def _handle_market_scan(self, parameters: dict) -> dict:
        """
        Handle a market scan operation.
        
        Args:
            parameters: Parameters for the market scan
            
        Returns:
            Results of the market scan
        """
        try:
            logger.info(f"Executing market scan with parameters: {parameters}")
            
            # Example implementation - would be connected to actual analysis tools
            crypto_symbol = parameters.get('crypto_symbol', 'BTC')
            scan_type = parameters.get('scan_type', 'technical')
            
            # Simulate market scan result
            result = {
                'timestamp': datetime.now().isoformat(),
                'crypto_symbol': crypto_symbol,
                'scan_type': scan_type,
                'market_conditions': 'bullish' if datetime.now().minute % 2 == 0 else 'bearish',
                'detected_patterns': ['double_bottom'] if datetime.now().minute % 3 == 0 else ['head_and_shoulders'],
                'recommendation': 'buy' if datetime.now().minute % 2 == 0 else 'sell'
            }
            
            logger.info(f"Market scan completed: {result}")
            return result
        except Exception as e:
            logger.error(f"Market scan failed: {str(e)}")
            raise
    
    def _handle_portfolio_rebalance(self, parameters: dict) -> dict:
        """
        Handle a portfolio rebalance operation.
        
        Args:
            parameters: Parameters for the portfolio rebalance
            
        Returns:
            Results of the portfolio rebalance
        """
        try:
            logger.info(f"Executing portfolio rebalance with parameters: {parameters}")
            
            # Example implementation - would be connected to actual portfolio management
            tolerance = parameters.get('tolerance', 5.0)  # Percentage tolerance for rebalancing
            
            # Simulate portfolio rebalance
            current_allocation = {
                'BTC': 45.0 + ((datetime.now().minute % 10) - 5),
                'ETH': 30.0 + ((datetime.now().minute % 6) - 3),
                'SOL': 15.0 + ((datetime.now().minute % 4) - 2),
                'USDT': 10.0 + ((datetime.now().minute % 2) - 1)
            }
            
            target_allocation = {
                'BTC': 45.0,
                'ETH': 30.0,
                'SOL': 15.0,
                'USDT': 10.0
            }
            
            # Calculate rebalance actions
            actions = []
            for asset, target in target_allocation.items():
                current = current_allocation[asset]
                deviation = abs(current - target)
                
                if deviation > tolerance:
                    action = 'buy' if current < target else 'sell'
                    amount = deviation / 100.0  # Convert percentage to decimal
                    actions.append({
                        'asset': asset,
                        'action': action,
                        'deviation': deviation,
                        'amount_percent': amount
                    })
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'current_allocation': current_allocation,
                'target_allocation': target_allocation,
                'tolerance': tolerance,
                'actions': actions,
                'rebalanced': len(actions) > 0
            }
            
            logger.info(f"Portfolio rebalance completed: {len(actions)} actions")
            return result
        except Exception as e:
            logger.error(f"Portfolio rebalance failed: {str(e)}")
            raise
    
    def _handle_trade_verification(self, parameters: dict) -> dict:
        """
        Handle a trade verification operation.
        
        Args:
            parameters: Parameters for the trade verification
            
        Returns:
            Results of the trade verification
        """
        try:
            logger.info(f"Executing trade verification with parameters: {parameters}")
            
            # Example implementation - would be connected to actual trading system
            lookback_days = parameters.get('lookback_days', 7)
            min_profit = parameters.get('min_profit', 1.0)  # Percentage
            
            # Simulate trade verification
            trade_history = [
                {
                    'id': f"trade_{i}",
                    'timestamp': (datetime.now() - timedelta(days=i % lookback_days, hours=i)).isoformat(),
                    'crypto': ['BTC', 'ETH', 'SOL'][i % 3],
                    'type': 'buy' if i % 2 == 0 else 'sell',
                    'position': 'long' if i % 3 != 2 else 'short',
                    'entry_price': 100 + (i * 10),
                    'exit_price': 100 + (i * 10) * (1 + (0.05 * (1 if i % 2 == 0 else -1))),
                    'position_size': 0.1 * (i % 5 + 1),
                    'profit_loss': 5.0 * (1 if i % 3 != 2 else -1)
                } for i in range(1, 6)
            ]
            
            # Filter and analyze trades
            verified_trades = []
            suspicious_trades = []
            total_profit = 0.0
            total_trades = len(trade_history)
            
            for trade in trade_history:
                profit = trade['profit_loss']
                total_profit += profit
                
                # Check for suspicious trades (very high profit or loss)
                if abs(profit) > 20.0:
                    trade['flag_reason'] = 'Unusual profit/loss amount'
                    suspicious_trades.append(trade)
                elif trade['position_size'] > 0.4:
                    trade['flag_reason'] = 'Unusually large position size'
                    suspicious_trades.append(trade)
                else:
                    verified_trades.append(trade)
            
            avg_profit = total_profit / total_trades if total_trades > 0 else 0
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'lookback_days': lookback_days,
                'total_trades': total_trades,
                'total_profit_percentage': total_profit,
                'average_profit_percentage': avg_profit,
                'verified_trades': len(verified_trades),
                'suspicious_trades': suspicious_trades,
                'meets_min_profit': avg_profit >= min_profit
            }
            
            logger.info(f"Trade verification completed: {len(suspicious_trades)} suspicious trades found")
            return result
        except Exception as e:
            logger.error(f"Trade verification failed: {str(e)}")
            raise
    
    def _handle_sentiment_analysis(self, parameters: dict) -> dict:
        """
        Handle a sentiment analysis operation.
        
        Args:
            parameters: Parameters for the sentiment analysis
            
        Returns:
            Results of the sentiment analysis
        """
        try:
            logger.info(f"Executing sentiment analysis with parameters: {parameters}")
            
            # Example implementation - would be connected to actual sentiment analysis tools
            crypto_symbol = parameters.get('crypto_symbol', 'BTC')
            sources = parameters.get('sources', ['twitter', 'reddit', 'news'])
            
            # Simulate sentiment analysis result
            sentiment_scores = {}
            for source in sources:
                # Generate pseudo-random but consistent scores based on time and source
                hour = datetime.now().hour
                base_score = ((hour + hash(source) % 10) % 10) / 10.0
                sentiment_scores[source] = round(-0.5 + base_score, 2)
            
            # Calculate average sentiment
            avg_sentiment = sum(sentiment_scores.values()) / len(sentiment_scores) if sentiment_scores else 0
            
            # Determine sentiment category
            if avg_sentiment > 0.3:
                sentiment_category = "Bullish"
            elif avg_sentiment > 0.1:
                sentiment_category = "Slightly Bullish"
            elif avg_sentiment > -0.1:
                sentiment_category = "Neutral"
            elif avg_sentiment > -0.3:
                sentiment_category = "Slightly Bearish"
            else:
                sentiment_category = "Bearish"
                
            # Generate trending topics/keywords
            keywords = [
                f"{crypto_symbol} rally",
                f"{crypto_symbol} analysis",
                "crypto regulation",
                "blockchain technology",
                f"{crypto_symbol} price prediction"
            ]
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'crypto_symbol': crypto_symbol,
                'sources_analyzed': sources,
                'sentiment_scores': sentiment_scores,
                'average_sentiment': avg_sentiment,
                'sentiment_category': sentiment_category,
                'trending_keywords': keywords,
                'recommended_action': "buy" if avg_sentiment > 0 else "sell" if avg_sentiment < -0.2 else "hold"
            }
            
            logger.info(f"Sentiment analysis completed: {sentiment_category} sentiment detected")
            return result
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            raise
            
    def cleanup(self):
        """Clean up resources when the tool is no longer needed."""
        self.manager.stop()
        

# Example of controlling recurring operations programmatically
def create_sample_recurring_operations():
    """
    Create sample recurring operations for demonstration purposes.
    """
    tool = RecurringOperationTool()
    
    # 1. Schedule a market scan every 30 minutes
    tool._run(
        operation_type='market_scan',
        schedule_type='interval',
        interval=30,
        parameters={
            'crypto_symbol': 'BTC',
            'scan_type': 'technical'
        }
    )
    
    # 2. Schedule daily portfolio rebalance
    tool._run(
        operation_type='portfolio_rebalance',
        schedule_type='daily',
        time_of_day='16:00',
        parameters={
            'tolerance': 5.0
        }
    )
    
    # 3. Schedule weekly trade verification
    tool._run(
        operation_type='trade_verification',
        schedule_type='weekly',
        day_of_week='monday',
        time_of_day='09:00',
        parameters={
            'lookback_days': 7,
            'min_profit': 1.5
        }
    )
    
    # 4. Schedule sentiment analysis twice a day
    tool._run(
        operation_type='sentiment_analysis',
        schedule_type='daily',
        time_of_day='09:30',
        parameters={
            'crypto_symbol': 'ETH',
            'sources': ['twitter', 'reddit', 'news']
        }
    )
    
    return "Sample recurring operations created"


if __name__ == "__main__":
    # Example of usage
    create_sample_recurring_operations() 