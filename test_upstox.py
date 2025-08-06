"""
Test script to verify Upstox API integration
"""
import asyncio
import os
from app.services.real_time_stock_service import RealTimeStockService

async def test_upstox_integration():
    """Test Upstox API for stock data retrieval"""
    
    # Initialize the service
    service = RealTimeStockService()
    
    # Test symbols
    test_symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ITC']
    
    print("Testing Upstox API Integration...")
    print("="*50)
    
    # Check if API keys are configured
    print(f"Upstox API Key configured: {'Yes' if service.upstox_api_key else 'No'}")
    print(f"Upstox Access Token configured: {'Yes' if service.upstox_access_token else 'No'}")
    print(f"Indian Stock API Key configured: {'Yes' if service.indian_stock_api_key else 'No'}")
    print()
    
    # Test real-time data
    print("Testing Real-time Data:")
    print("-"*30)
    
    for symbol in test_symbols:
        try:
            print(f"Fetching data for {symbol}...")
            data = await service.get_real_time_data(symbol)
            
            if data:
                print(f"✓ {symbol}: ₹{data.close:.2f} (Change: {data.change:+.2f}, {data.change_percent:+.2f}%)")
            else:
                print(f"✗ {symbol}: No data available")
        except Exception as e:
            print(f"✗ {symbol}: Error - {e}")
    
    print()
    
    # Test historical data
    print("Testing Historical Data (1 month):")
    print("-"*35)
    
    for symbol in ['RELIANCE', 'TCS']:  # Test fewer symbols for historical data
        try:
            print(f"Fetching historical data for {symbol}...")
            data = await service.get_historical_data(symbol, "1mo")
            
            if data and data.data:
                print(f"✓ {symbol}: {len(data.data)} data points retrieved")
                # Show last few data points
                recent_data = data.data[-3:]
                for point in recent_data:
                    print(f"   {point.timestamp.strftime('%Y-%m-%d')}: ₹{point.close:.2f}")
            else:
                print(f"✗ {symbol}: No historical data available")
        except Exception as e:
            print(f"✗ {symbol}: Error - {e}")
    
    # Close the session
    await service.close_session()
    
    print()
    print("Test completed!")

if __name__ == "__main__":
    asyncio.run(test_upstox_integration())
