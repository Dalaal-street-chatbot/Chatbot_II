import React, { useEffect, useRef, useState, useCallback } from 'react';
import { createChart, IChartApi, ISeriesApi, CandlestickData, CandlestickSeries } from 'lightweight-charts';
import { API_ENDPOINTS, buildApiUrl } from '../config/api';
import './TradingChart.css';

interface TradingChartProps {
  symbol?: string;
  onSymbolChange?: (symbol: string) => void;
}

interface ChartDataPoint {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

// Upstox API Response Interfaces
interface UpstoxQuote {
  last_price: number;
  ohlc: {
    open: number;
    high: number;
    low: number;
    close: number;
  };
  volume: number;
  net_change: number;
  percent_change: number;
}

const TradingChart: React.FC<TradingChartProps> = ({ symbol = 'RELIANCE', onSymbolChange }) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  
  const [currentSymbol, setCurrentSymbol] = useState(symbol);
  const [isLoading, setIsLoading] = useState(false);
  const [currentPrice, setCurrentPrice] = useState<number | null>(null);
  const [priceChange, setPriceChange] = useState<{ value: number; percent: number } | null>(null);
  const [currentTimeframe, setCurrentTimeframe] = useState('1D');
  const [dataSource, setDataSource] = useState<'upstox' | 'api' | 'sample'>('api');

  const symbols = [
    // Market Indices
    'NIFTY50', 'SENSEX', 'BANKNIFTY', 
    // Individual Stocks
    'RELIANCE', 'TCS', 'INFY', 'HDFC', 'ITC', 'SBIN', 'BAJFINANCE', 'LT'
  ];
  const timeframes = ['15m', '1h', '4h', '1D', '1W'];

  // Define loadSampleData first so we can reference it in the dependency array
  const loadSampleData = useCallback((selectedSymbol: string, selectedTimeframe: string) => {
    // Generate sample data that's consistent with the Upstox API format
    const sampleData: ChartDataPoint[] = [];
    
    // Stock-specific and index-specific base prices for more realistic data
    const baseStockPrices: {[key: string]: number} = {
      // Market Indices
      'NIFTY50': 22500,
      'SENSEX': 74000,
      'BANKNIFTY': 48000,
      // Individual Stocks
      'RELIANCE': 2500,
      'TCS': 3200,
      'INFY': 1400,
      'HDFC': 2700,
      'ITC': 450,
      'SBIN': 550,
      'BAJFINANCE': 6500,
      'LT': 2800,
    };
    
    const basePrice = baseStockPrices[selectedSymbol] || 2500 + Math.random() * 1000;
    let currentPrice = basePrice;
    
    // Generate many more data points for better visualization and filling the chart
    // The number of data points depends on the selected timeframe
    const dataPoints = 
      selectedTimeframe === '15m' ? 300 :   // 300 points for 15 min charts
      selectedTimeframe === '1h' ? 250 :    // 250 points for hourly charts
      selectedTimeframe === '4h' ? 200 :    // 200 points for 4-hour charts 
      selectedTimeframe === '1D' ? 180 :    // 180 points for daily charts
      selectedTimeframe === '1W' ? 150 : 120; // 150 points for weekly charts, default to 120
    
    for (let i = 0; i < dataPoints; i++) {
      // More granular timeframes for intraday charts
      const date = new Date();
      const isIntraday = ['15m', '1h', '4h'].includes(selectedTimeframe);
      
      if (isIntraday) {
        if (selectedTimeframe === '15m') {
          // For 15-minute charts, use minutes
          date.setMinutes(date.getMinutes() - ((dataPoints - 1) - i) * 15);
        } else if (selectedTimeframe === '1h') {
          // For hourly charts, use hours
          date.setHours(date.getHours() - ((dataPoints - 1) - i));
        } else {
          // For 4h charts
          date.setHours(date.getHours() - ((dataPoints - 1) - i) * 4);
        }
      } else if (selectedTimeframe === '1D') {
        // For daily charts, use days
        date.setDate(date.getDate() - ((dataPoints - 1) - i));
      } else {
        // For weekly charts, use weeks
        date.setDate(date.getDate() - ((dataPoints - 1) - i) * 7);
      }
      
      // More realistic price movements
      const volatility = basePrice * 0.015; // 1.5% volatility
      
      // Enhanced simulation with trends and volatility based on timeframe
      // Different timeframes have different volatility characteristics
      const timeframeVolatility = 
        selectedTimeframe === '15m' ? volatility * 0.6 : 
        selectedTimeframe === '1h' ? volatility * 0.8 : 
        selectedTimeframe === '4h' ? volatility * 1.0 : 
        selectedTimeframe === '1D' ? volatility * 1.2 : 
        volatility * 1.5; // Weekly has highest volatility
      
      // Create some trends to make the chart more realistic
      const trendComponent = Math.sin(i * 0.05) * timeframeVolatility * 1.5;
      const cycleComponent = Math.sin(i * 0.2) * timeframeVolatility * 0.8;
      const noiseComponent = (Math.random() - 0.5) * timeframeVolatility;
      
      // Apply trend to current price
      currentPrice = i === 0 ? basePrice : currentPrice + trendComponent * 0.1;
      
      // Create realistic OHLC values
      const open = currentPrice;
      const direction = Math.random() > 0.5 ? 1 : -1;
      const range = timeframeVolatility * (0.5 + Math.random() * 0.5);
      const close = open + (direction * range) + cycleComponent + noiseComponent;
      const high = Math.max(open, close) + (Math.random() * timeframeVolatility * 0.5);
      const low = Math.min(open, close) - (Math.random() * timeframeVolatility * 0.5);
      
      // More realistic volume based on price
      const avgVolume = Math.round(basePrice > 3000 ? 500000 : 1000000);
      const volume = Math.floor(Math.random() * avgVolume) + avgVolume;
      
      let timeString: string;
      if (isIntraday) {
        // For intraday, use timestamp in seconds
        timeString = Math.floor(date.getTime() / 1000).toString();
      } else if (selectedTimeframe === '1D') {
        // For daily, use YYYY-MM-DD format
        timeString = date.toISOString().split('T')[0];
      } else {
        // For weekly, use timestamp in seconds (needed for proper time scale display)
        // Weekly charts require Unix timestamp to display properly in Lightweight Charts
        timeString = Math.floor(date.getTime() / 1000).toString();
      }
      
      sampleData.push({
        time: timeString,
        open,
        high,
        low,
        close,
        volume
      });
      
      currentPrice = close;
    }
    
    if (candlestickSeriesRef.current) {
      const candlestickData: CandlestickData[] = sampleData.map(item => ({
        time: item.time,
        open: item.open,
        high: item.high,
        low: item.low,
        close: item.close,
      }));
      candlestickSeriesRef.current.setData(candlestickData);
    }
    
    const lastDataPoint = sampleData[sampleData.length - 1];
    setCurrentPrice(lastDataPoint.close);
    const change = lastDataPoint.close - lastDataPoint.open;
    setPriceChange({
      value: change,
      percent: (change / lastDataPoint.open) * 100
    });
    
    console.log('Loaded sample data for', selectedSymbol);
    setDataSource('sample');
  }, []);

  const loadChartData = useCallback(async (selectedSymbol: string, selectedTimeframe: string) => {
    setIsLoading(true);
    try {
      // First try using backend API which prioritizes Upstox data
      const apiUrl = buildApiUrl(API_ENDPOINTS.CHART_DATA, {
        symbol: selectedSymbol,
        timeframe: selectedTimeframe
      });
      
      const response = await fetch(apiUrl);
      if (response.ok) {
        const data = await response.json();
        if (candlestickSeriesRef.current && data.candlestick_data) {
          const candlestickData: CandlestickData[] = data.candlestick_data.map((item: any) => ({
            time: item.time,
            open: item.open,
            high: item.high,
            low: item.low,
            close: item.close,
          }));
          candlestickSeriesRef.current.setData(candlestickData);
        }
        
        setCurrentPrice(data.current_price);
        setPriceChange({
          value: data.change || 0, 
          percent: data.change_percent || 0
        });
        
        console.log('Loaded chart data from API (with Upstox as primary source)');
        setDataSource('upstox');
      } else {
        console.warn('API request failed, falling back to sample data');
        // Fallback to sample data
        loadSampleData(selectedSymbol, selectedTimeframe);
      }
    } catch (error) {
      console.error('Error loading chart data:', error);
      loadSampleData(selectedSymbol, selectedTimeframe);
    }
    setIsLoading(false);
  }, [loadSampleData]);



  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Create chart with enhanced options
    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 500,
      layout: {
        background: { color: '#1a1a1a' },
        textColor: '#ffffff',
      },
      grid: {
        vertLines: { color: '#334155' },
        horzLines: { color: '#334155' },
      },
      crosshair: {
        mode: 1,
      },
      rightPriceScale: {
        borderColor: '#485563',
        autoScale: true,
        scaleMargins: {
          top: 0.1,
          bottom: 0.2,
        },
      },
      timeScale: {
        borderColor: '#485563',
        timeVisible: true,
        secondsVisible: false,
        tickMarkFormatter: (time: number) => {
          const date = new Date(time * 1000);
          return date.getHours().toString().padStart(2, '0') + ':' + 
                 date.getMinutes().toString().padStart(2, '0');
        },
      },
    });

    // Create candlestick series using the lightweight-charts API
    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#4ade80',
      downColor: '#ef4444',
      borderDownColor: '#ef4444',
      borderUpColor: '#4ade80',
      wickDownColor: '#ef4444',
      wickUpColor: '#4ade80',
    });

    chartRef.current = chart;
    candlestickSeriesRef.current = candlestickSeries;

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };

    window.addEventListener('resize', handleResize);

    // Load initial data using the Upstox data source
    loadChartData(currentSymbol, currentTimeframe);

    // Setup live data refresh interval (every 30 seconds)
    console.log('TradingChart initialized with symbol:', currentSymbol);
    
    // Set up auto-refresh for live data
    const refreshInterval = setInterval(() => {
      // Only auto-refresh when viewing current timeframes (1h or lower)
      if (['15m', '1h'].includes(currentTimeframe)) {
        console.log(`Auto-refreshing ${currentSymbol} data...`);
        loadChartData(currentSymbol, currentTimeframe);
      }
    }, 30000); // Every 30 seconds

    return () => {
      window.removeEventListener('resize', handleResize);
      clearInterval(refreshInterval);
      chart.remove();
    };
  }, [currentSymbol, currentTimeframe, loadChartData]);

  const handleSymbolChange = (newSymbol: string) => {
    setCurrentSymbol(newSymbol);
    if (onSymbolChange) {
      onSymbolChange(newSymbol);
    }
  };

  const handleTimeframeChange = (tf: string) => {
    setCurrentTimeframe(tf);
    loadChartData(currentSymbol, tf);
  };

  return (
    <div className="trading-chart">
      <div className="chart-header">
        <div className="chart-controls">
          <div className="symbol-selector">
            <label htmlFor="symbol-select">Symbol:</label>
            <select
              id="symbol-select"
              value={currentSymbol}
              onChange={(e) => handleSymbolChange(e.target.value)}
              className="symbol-select"
            >
              <optgroup label="Market Indices">
                {symbols.filter(sym => ['NIFTY50', 'SENSEX', 'BANKNIFTY'].includes(sym)).map((sym) => (
                  <option key={sym} value={sym}>
                    {sym}
                  </option>
                ))}
              </optgroup>
              <optgroup label="Stocks">
                {symbols.filter(sym => !['NIFTY50', 'SENSEX', 'BANKNIFTY'].includes(sym)).map((sym) => (
                  <option key={sym} value={sym}>
                    {sym}
                  </option>
                ))}
              </optgroup>
            </select>
          </div>
          
          <div className="timeframe-selector">
            {timeframes.map((tf) => (
              <button
                key={tf}
                className={`timeframe-btn${currentTimeframe === tf ? ' active' : ''}`}
                onClick={() => handleTimeframeChange(tf)}
              >
                {tf}
              </button>
            ))}
          </div>
        </div>
        
        <div className="price-info">
          {currentPrice && (
            <>
              <div className="current-price">
                ₹{currentPrice.toFixed(2)}
                {dataSource === 'upstox' && (
                  <span className="data-source upstox">• Upstox</span>
                )}
              </div>
              {priceChange && (
                <div className={`price-change ${priceChange.value >= 0 ? 'positive' : 'negative'}`}>
                  {priceChange.value >= 0 ? '+' : ''}₹{priceChange.value.toFixed(2)} 
                  ({priceChange.value >= 0 ? '+' : ''}{priceChange.percent.toFixed(2)}%)
                </div>
              )}
              <div className="chart-info">
                {currentTimeframe} • {
                  currentTimeframe === '15m' ? '300+ data points' :
                  currentTimeframe === '1h' ? '250+ data points' :
                  currentTimeframe === '4h' ? '200+ data points' :
                  currentTimeframe === '1D' ? '180+ data points' :
                  '150+ data points'
                }
              </div>
            </>
          )}
        </div>
      </div>
      
      <div className="chart-container">
        {isLoading && (
          <div className="loading-overlay">
            <div className="loading-spinner"></div>
            <p>Loading chart data...</p>
          </div>
        )}
        <div ref={chartContainerRef} className="chart" />
      </div>
    </div>
  );
};

export default TradingChart;
