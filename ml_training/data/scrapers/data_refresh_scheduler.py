#!/usr/bin/env python3
"""
Data Refresh Scheduler - Periodically refreshes financial data from scrapers

This script:
1. Runs scrapers at scheduled intervals
2. Updates cached data
3. Logs refresh activities and errors
4. Monitors system health and performance

Usage:
    python data_refresh_scheduler.py [--daemon] [--log-level LEVEL]

Options:
    --daemon         Run as a background daemon
    --log-level      Set logging level (DEBUG, INFO, WARNING, ERROR)
"""

import os
import sys
import time
import argparse
import logging
import asyncio
import json
import signal
import datetime
import traceback
import psutil
from typing import Dict, Any, List
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Define log file path
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'data_refresh.log')

# Configure rotating file handler
from logging.handlers import RotatingFileHandler

# Create logger
logger = logging.getLogger("DataRefreshScheduler")
logger.setLevel(logging.INFO)

# Create handlers
console_handler = logging.StreamHandler()
file_handler = RotatingFileHandler(
    LOG_FILE, 
    maxBytes=10485760,  # 10MB
    backupCount=5
)

# Create formatters
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Import scrapers
try:
    from money_control_scraper import MoneyControlScraper
    from google_finance_scraper import GoogleFinanceScraper
    from cnbc_tv18_scraper import CnbcTv18Scraper
    from financial_news_aggregator import FinancialNewsAggregator
except ImportError as e:
    # Add the parent directory to sys.path
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Now try again
    from money_control_scraper import MoneyControlScraper
    from google_finance_scraper import GoogleFinanceScraper
    from cnbc_tv18_scraper import CnbcTv18Scraper
    from financial_news_aggregator import FinancialNewsAggregator

# Cache directory
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

class DataRefreshScheduler:
    """Scheduler for refreshing financial data"""
    
    def __init__(self):
        """Initialize the scheduler"""
        self.running = False
        self.tasks = {}
        self.last_run = {}
        self.data_cache = {}
        
        # Initialize scrapers
        self.money_control = MoneyControlScraper()
        self.google_finance = GoogleFinanceScraper()
        self.cnbc_tv18 = CnbcTv18Scraper()
        self.news_aggregator = FinancialNewsAggregator()
        
        # Schedule configuration (in seconds)
        self.schedules = {
            "top_news": 1800,           # 30 minutes
            "market_data": 900,         # 15 minutes
            "stock_data": 1200,         # 20 minutes
            "market_sentiment": 1800,   # 30 minutes
            "expert_opinions": 3600,    # 60 minutes
            "market_alerts": 600,       # 10 minutes
            "full_dashboard": 3600      # 60 minutes
        }
        
        # Load existing cache if available
        self._load_cache()
        
        logger.info("Data Refresh Scheduler initialized")
    
    def _load_cache(self):
        """Load existing cache files"""
        for task_name in self.schedules.keys():
            cache_file = os.path.join(CACHE_DIR, f"{task_name}.json")
            try:
                if os.path.exists(cache_file):
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                        self.data_cache[task_name] = data
                        self.last_run[task_name] = data.get("timestamp", 0)
                        logger.info(f"Loaded cache for {task_name}")
            except Exception as e:
                logger.error(f"Error loading cache for {task_name}: {e}")
    
    def _save_cache(self, task_name: str, data: Dict[str, Any]):
        """Save data to cache file
        
        Args:
            task_name: Name of the task/data type
            data: Data to cache
        """
        cache_file = os.path.join(CACHE_DIR, f"{task_name}.json")
        try:
            # Add timestamp
            if isinstance(data, dict):
                data["timestamp"] = time.time()
                data["refresh_time"] = datetime.datetime.now().isoformat()
            
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            self.data_cache[task_name] = data
            self.last_run[task_name] = time.time()
            logger.info(f"Saved cache for {task_name}")
        except Exception as e:
            logger.error(f"Error saving cache for {task_name}: {e}")
    
    async def refresh_top_news(self):
        """Refresh top financial news"""
        try:
            logger.info("Refreshing top financial news...")
            news = await self.news_aggregator.get_all_top_news(limit=10)
            self._save_cache("top_news", {
                "articles": news,
                "total_results": len(news)
            })
            logger.info(f"Refreshed top news: {len(news)} articles")
            return True
        except Exception as e:
            logger.error(f"Error refreshing top news: {e}")
            return False
    
    async def refresh_market_data(self):
        """Refresh market data"""
        try:
            logger.info("Refreshing market data...")
            
            # Get market trends from Google Finance
            market_trends = self.google_finance.get_market_trends()
            
            # Get major indices
            nifty_data = self.google_finance.get_index_data("NIFTY")
            sensex_data = self.google_finance.get_index_data("SENSEX")
            
            self._save_cache("market_data", {
                "trends": market_trends,
                "indices": {
                    "NIFTY": nifty_data,
                    "SENSEX": sensex_data
                }
            })
            
            logger.info("Refreshed market data")
            return True
        except Exception as e:
            logger.error(f"Error refreshing market data: {e}")
            return False
    
    async def refresh_stock_data(self):
        """Refresh stock data for major stocks"""
        try:
            logger.info("Refreshing stock data...")
            
            major_stocks = [
                "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", 
                "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK"
            ]
            
            stock_data = {}
            for symbol in major_stocks:
                try:
                    data = self.google_finance.get_stock_data(symbol)
                    stock_data[symbol] = data
                    logger.info(f"Refreshed stock data for {symbol}")
                    # Add a small delay to avoid rate limiting
                    await asyncio.sleep(1)
                except Exception as stock_error:
                    logger.warning(f"Error refreshing stock data for {symbol}: {stock_error}")
            
            self._save_cache("stock_data", stock_data)
            logger.info(f"Refreshed stock data for {len(stock_data)} stocks")
            return True
        except Exception as e:
            logger.error(f"Error refreshing stock data: {e}")
            return False
    
    async def refresh_market_sentiment(self):
        """Refresh market sentiment data"""
        try:
            logger.info("Refreshing market sentiment...")
            sentiment = await self.news_aggregator.get_market_sentiment()
            self._save_cache("market_sentiment", sentiment)
            logger.info("Refreshed market sentiment")
            return True
        except Exception as e:
            logger.error(f"Error refreshing market sentiment: {e}")
            return False
    
    async def refresh_expert_opinions(self):
        """Refresh expert opinions"""
        try:
            logger.info("Refreshing expert opinions...")
            opinions = self.cnbc_tv18.get_expert_opinions(limit=5)
            self._save_cache("expert_opinions", {
                "opinions": opinions,
                "total_results": len(opinions)
            })
            logger.info(f"Refreshed expert opinions: {len(opinions)} opinions")
            return True
        except Exception as e:
            logger.error(f"Error refreshing expert opinions: {e}")
            return False
    
    async def refresh_market_alerts(self):
        """Refresh market alerts"""
        try:
            logger.info("Refreshing market alerts...")
            alerts = self.cnbc_tv18.get_market_alerts()
            self._save_cache("market_alerts", {
                "alerts": alerts,
                "total_results": len(alerts)
            })
            logger.info(f"Refreshed market alerts: {len(alerts)} alerts")
            return True
        except Exception as e:
            logger.error(f"Error refreshing market alerts: {e}")
            return False
    
    async def refresh_full_dashboard(self):
        """Refresh all dashboard data in a single operation"""
        try:
            logger.info("Refreshing full market dashboard...")
            
            # Run all these tasks in parallel
            top_news_task = asyncio.create_task(self.news_aggregator.get_all_top_news(limit=5))
            sentiment_task = asyncio.create_task(self.news_aggregator.get_market_sentiment())
            
            # These are synchronous, so we'll run them directly
            market_trends = self.google_finance.get_market_trends()
            expert_opinions = self.cnbc_tv18.get_expert_opinions(limit=3)
            alerts = self.cnbc_tv18.get_market_alerts()
            
            # Major indices
            nifty_data = self.google_finance.get_index_data("NIFTY")
            sensex_data = self.google_finance.get_index_data("SENSEX")
            
            # Wait for async tasks to complete
            top_news = await top_news_task
            sentiment = await sentiment_task
            
            dashboard = {
                "timestamp": time.time(),
                "refresh_time": datetime.datetime.now().isoformat(),
                "top_news": top_news,
                "market_sentiment": sentiment,
                "market_trends": market_trends,
                "indices": {
                    "NIFTY": nifty_data,
                    "SENSEX": sensex_data
                },
                "expert_opinions": expert_opinions,
                "market_alerts": alerts
            }
            
            self._save_cache("full_dashboard", dashboard)
            logger.info("Refreshed full market dashboard")
            return True
        except Exception as e:
            logger.error(f"Error refreshing full dashboard: {e}")
            return False
    
    async def check_schedule(self):
        """Check if any tasks need to be run based on schedule"""
        current_time = time.time()
        
        tasks = []
        
        for task_name, interval in self.schedules.items():
            last_run = self.last_run.get(task_name, 0)
            if current_time - last_run >= interval:
                logger.info(f"Scheduling task: {task_name}")
                
                if task_name == "top_news":
                    tasks.append(self.refresh_top_news())
                elif task_name == "market_data":
                    tasks.append(self.refresh_market_data())
                elif task_name == "stock_data":
                    tasks.append(self.refresh_stock_data())
                elif task_name == "market_sentiment":
                    tasks.append(self.refresh_market_sentiment())
                elif task_name == "expert_opinions":
                    tasks.append(self.refresh_expert_opinions())
                elif task_name == "market_alerts":
                    tasks.append(self.refresh_market_alerts())
                elif task_name == "full_dashboard":
                    tasks.append(self.refresh_full_dashboard())
        
        if tasks:
            await asyncio.gather(*tasks)
    
    async def run_once(self):
        """Run one cycle of the scheduler"""
        await self.check_schedule()
    
    async def health_check(self):
        """Perform health check and log system metrics"""
        try:
            # Log system metrics
            await log_system_metrics()
            
            # Check cache health
            cache_health = {
                "timestamp": datetime.datetime.now().isoformat(),
                "cache_items": len(self.data_cache),
                "last_run_stats": {}
            }
            
            # Check if any task hasn't run recently
            current_time = time.time()
            for task_name, interval in self.schedules.items():
                last_run = self.last_run.get(task_name, 0)
                time_since_last_run = current_time - last_run
                
                cache_health["last_run_stats"][task_name] = {
                    "last_run": datetime.datetime.fromtimestamp(last_run).isoformat() if last_run > 0 else "never",
                    "seconds_since_last_run": time_since_last_run,
                    "expected_interval": interval,
                    "overdue": time_since_last_run > interval * 2 if last_run > 0 else True
                }
                
                # Alert if a task is seriously overdue
                if last_run > 0 and time_since_last_run > interval * 2:
                    logger.warning(f"Task {task_name} is overdue. Last run: {time_since_last_run:.2f}s ago, interval: {interval}s")
            
            # Save health check to file
            health_file = os.path.join(LOG_DIR, 'scheduler_health.json')
            with open(health_file, 'w') as f:
                json.dump(cache_health, f, indent=2)
                
            return True
        except Exception as e:
            logger.error(f"Error in health check: {e}")
            return False
    
    async def run_forever(self):
        """Run the scheduler indefinitely with monitoring"""
        self.running = True
        
        def handle_stop(sig, frame):
            logger.info(f"Received signal {sig}, shutting down...")
            self.running = False
        
        signal.signal(signal.SIGINT, handle_stop)
        signal.signal(signal.SIGTERM, handle_stop)
        signal.signal(signal.SIGHUP, handle_stop)  # Handle SIGHUP for service reloads
        
        logger.info("Starting data refresh scheduler...")
        
        # Track consecutive failures for backoff strategy
        consecutive_failures = 0
        health_check_interval = 300  # 5 minutes
        last_health_check = 0
        
        while self.running:
            current_time = time.time()
            try:
                # Run scheduled tasks
                await self.run_once()
                consecutive_failures = 0  # Reset failure counter on success
                
                # Run health check at regular intervals
                if current_time - last_health_check >= health_check_interval:
                    await self.health_check()
                    last_health_check = current_time
                
                # Sleep for 60 seconds before checking schedule again
                for _ in range(60):
                    if not self.running:
                        break
                    await asyncio.sleep(1)
                    
            except Exception as e:
                consecutive_failures += 1
                backoff_time = min(60 * 2 ** consecutive_failures, 3600)  # Exponential backoff, max 1 hour
                
                logger.error(f"Error in scheduler loop (attempt {consecutive_failures}): {e}")
                logger.error(traceback.format_exc())
                logger.warning(f"Backing off for {backoff_time} seconds before retrying")
                
                # Still do health check even if main loop fails
                if current_time - last_health_check >= health_check_interval:
                    await self.health_check()
                    last_health_check = current_time
                
                # Sleep with backoff strategy
                await asyncio.sleep(backoff_time)
        
        logger.info("Data refresh scheduler stopped")

async def log_system_metrics():
    """Log system metrics for monitoring"""
    try:
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        metrics = {
            "timestamp": datetime.datetime.now().isoformat(),
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_mb": memory.available / (1024 * 1024),
            "disk_percent": disk.percent,
            "disk_free_gb": disk.free / (1024 * 1024 * 1024)
        }
        
        # Log metrics
        logger.info(f"System metrics: CPU: {cpu_percent}%, Memory: {memory.percent}%, Disk: {disk.percent}%")
        
        # Save metrics to file
        metrics_file = os.path.join(LOG_DIR, 'system_metrics.json')
        try:
            # Load existing metrics
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    existing_metrics = json.load(f)
                    # Keep only the last 100 entries
                    if len(existing_metrics) >= 100:
                        existing_metrics = existing_metrics[-99:]
            else:
                existing_metrics = []
                
            # Append new metrics
            existing_metrics.append(metrics)
            
            # Save updated metrics
            with open(metrics_file, 'w') as f:
                json.dump(existing_metrics, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving system metrics: {e}")
            
    except Exception as e:
        logger.error(f"Error collecting system metrics: {e}")

async def run_scheduler(daemon_mode=False, refresh_all=False):
    """Run the data refresh scheduler
    
    Args:
        daemon_mode: Whether to run in daemon mode (background)
        refresh_all: Whether to refresh all data immediately
    """
    scheduler = DataRefreshScheduler()
    
    if refresh_all:
        logger.info("Refreshing all data immediately")
        await scheduler.refresh_full_dashboard()
        await scheduler.refresh_stock_data()
        logger.info("Full refresh completed")
    
    if daemon_mode:
        logger.info("Running in daemon mode")
        
        # Write PID to file for systemd management
        pid_file = os.path.join(LOG_DIR, 'data_refresh.pid')
        with open(pid_file, 'w') as f:
            f.write(str(os.getpid()))
        
        # Log initial system metrics
        await log_system_metrics()
        
        # Start the scheduler
        await scheduler.run_forever()
    else:
        logger.info("Running refresh once")
        await scheduler.run_once()
        await log_system_metrics()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Data Refresh Scheduler")
    parser.add_argument('--daemon', action='store_true', help="Run as daemon")
    parser.add_argument('--refresh-all', action='store_true', help="Refresh all data immediately")
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                      default='INFO', help="Set the logging level")
    
    args = parser.parse_args()
    
    # Set logging level based on argument
    log_level = getattr(logging, args.log_level)
    logger.setLevel(log_level)
    console_handler.setLevel(log_level)
    file_handler.setLevel(log_level)
    
    # Record start in log
    logger.info("=" * 50)
    logger.info(f"Starting Data Refresh Scheduler (PID: {os.getpid()})")
    logger.info(f"Log level: {args.log_level}")
    logger.info(f"Daemon mode: {args.daemon}")
    logger.info(f"Refresh all: {args.refresh_all}")
    logger.info("=" * 50)
    
    try:
        asyncio.run(run_scheduler(args.daemon, args.refresh_all))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
