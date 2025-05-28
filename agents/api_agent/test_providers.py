"""
Test script to validate the multi-provider financial data implementation
This script demonstrates fetching data from multiple providers with fallback
"""

import asyncio
import sys
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Setup proper logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to import modules
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the client and models
from agents.api_agent.client import (
    FinancialDataClient, fetch_current_price, fetch_historical,
    DataProvider, APIClientError, _client
)
from agents.api_agent.models import (
    MultiProviderPriceRequest, DataProvider as ModelDataProvider
)
from agents.api_agent.config import settings


async def test_individual_providers():
    """Test each provider individually for current price"""
    providers = list(_client.providers.keys())
    symbol = "AAPL"  # Apple Inc. as a commonly available test symbol
    
    logger.info(f"Testing {len(providers)} individual providers for {symbol}")
    
    for provider in providers:
        try:
            logger.info(f"Testing {provider}...")
            result = await fetch_current_price(symbol, provider)
            logger.info(f"✅ {provider}: {result['symbol']} = ${result['price']:.2f} [{result['timestamp']}]")
        except Exception as e:
            logger.error(f"❌ {provider} failed: {str(e)}")


async def test_fallback_mechanism():
    """Test the fallback mechanism with an intentionally invalid provider first"""
    symbol = "MSFT"  # Microsoft as test symbol
    
    # Create a custom provider order with an invalid one first
    test_providers = ["invalid_provider"] + list(_client.providers.keys())
    
    logger.info(f"Testing fallback mechanism for {symbol}")
    logger.info(f"Provider order: {test_providers}")
    
    try:
        # Should fallback to a valid provider
        result = await _client.fetch_current_price(symbol, test_providers[0])
        logger.info(f"✅ Fallback succeeded: {result['symbol']} = ${result['price']:.2f} from {result['provider']}")
    except Exception as e:
        logger.error(f"❌ Fallback failed: {str(e)}")


async def test_historical_data():
    """Test fetching historical data from different providers"""
    symbol = "GOOGL"  # Alphabet Inc. as test symbol
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    logger.info(f"Testing historical data for {symbol} from {start_date.date()} to {end_date.date()}")
    
    for provider in _client.providers.keys():
        try:
            logger.info(f"Testing {provider} for historical data...")
            result = await fetch_historical(symbol, start_date, end_date, provider)
            logger.info(f"✅ {provider}: Got {len(result['timeseries'])} days of data")
            
            # Show the first and last data points
            if result['timeseries']:
                first = result['timeseries'][0]
                last = result['timeseries'][-1]
                logger.info(f"   First: {first['date']} - Open: ${first['open']:.2f}, Close: ${first['close']:.2f}")
                logger.info(f"   Last: {last['date']} - Open: ${last['open']:.2f}, Close: ${last['close']:.2f}")
        except Exception as e:
            logger.error(f"❌ {provider} historical data failed: {str(e)}")


async def test_multi_provider_consensus():
    """Test the multi-provider consensus feature"""
    symbol = "AMZN"  # Amazon as test symbol
    
    logger.info(f"Testing multi-provider consensus for {symbol}")
    
    # Simulate the multi-provider endpoint
    providers_to_try = [p for p in _client.providers.keys()]
    prices = []
    
    for provider in providers_to_try:
        try:
            result = await fetch_current_price(symbol, provider)
            logger.info(f"Provider {provider}: ${result['price']:.2f}")
            prices.append(result['price'])
        except Exception as e:
            logger.error(f"Provider {provider} failed: {str(e)}")
    
    if prices:
        # Calculate simple consensus (mean)
        import statistics
        consensus = statistics.mean(prices)
        logger.info(f"Consensus price for {symbol}: ${consensus:.2f}")
        
        # Show variance
        if len(prices) > 1:
            variance = statistics.variance(prices)
            logger.info(f"Price variance: {variance:.4f}")
    else:
        logger.error("No valid prices obtained from any provider")


async def main():
    """Run all tests sequentially"""
    logger.info("🚀 Starting multi-provider financial data API tests")
    logger.info(f"Available providers: {list(_client.providers.keys())}")
    
    await test_individual_providers()
    logger.info("-" * 40)
    
    await test_fallback_mechanism()
    logger.info("-" * 40)
    
    await test_historical_data()
    logger.info("-" * 40)
    
    await test_multi_provider_consensus()
    
    logger.info("✨ All tests completed")


if __name__ == "__main__":
    asyncio.run(main())
