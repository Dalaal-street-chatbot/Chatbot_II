import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { API_ENDPOINTS } from '../config/api';
import './ChatWindow.css';

interface ChatWindowProps {
  onSymbolMention?: (symbol: string) => void;
}

interface Message {
  id: number;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  isTyping?: boolean;
}

interface ChatResponse {
  response: string;
  session_id?: string;
  intent?: string;
  response_type?: string;
  stock_data?: any;
  news_items?: any[];
  sentiment_data?: any;
  portfolio?: any;
}

const ChatWindow: React.FC<ChatWindowProps> = ({ onSymbolMention }) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      text: "Hello! I'm your Dalaal Street assistant. I can help you with stock prices, market news, portfolio management, and financial advice for Indian markets. What would you like to know?",
      sender: 'bot',
      timestamp: new Date()
    }
  ]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string>('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async (text: string) => {
    if (!text.trim()) return;

    // Add user message
    const userMessage: Message = {
      id: Date.now(),
      text: text.trim(),
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);

    try {
      // Try different API endpoints based on keywords
      let apiEndpoint: string = API_ENDPOINTS.CHAT;
      
      // Check for Google Cloud/Dialogflow keywords
      const googleKeywords = ['portfolio', 'analyze', 'sentiment', 'compare', 'news', 'stock'];
      const hasGoogleKeywords = googleKeywords.some(keyword => 
        text.toLowerCase().includes(keyword)
      );

      if (hasGoogleKeywords) {
        apiEndpoint = API_ENDPOINTS.GOOGLE_CLOUD_CHAT;
      }

      const response = await axios.post(apiEndpoint, {
        message: text.trim(),
        session_id: sessionId
      }, {
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        }
      });

      const data: ChatResponse = response.data;
      
      // Update session ID if provided
      if (data.session_id && !sessionId) {
        setSessionId(data.session_id);
      }

      // Check for stock symbols in user message or bot response and notify parent
      if (onSymbolMention) {
        const symbols = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR', 
                        'ICICIBANK', 'BHARTIARTL', 'SBIN', 'LT', 'WIPRO'];
        const messageText = text.toUpperCase();
        const responseText = data.response?.toUpperCase() || '';
        
        for (const symbol of symbols) {
          if (messageText.includes(symbol) || responseText.includes(symbol)) {
            onSymbolMention(symbol);
            break;
          }
        }
      }

      // Format response based on type
      let botResponseText = data.response || 'Sorry, I couldn\'t process that request.';
      
      // Enhanced formatting for different response types
      if (data.response_type === 'enhanced_stock' && data.stock_data) {
        botResponseText = formatStockResponse(data);
      } else if (data.response_type === 'news' && data.news_items) {
        botResponseText = formatNewsResponse(data);
      } else if (data.response_type === 'sentiment' && data.sentiment_data) {
        botResponseText = formatSentimentResponse(data);
      } else if (data.response_type === 'portfolio' && data.portfolio) {
        botResponseText = formatPortfolioResponse(data);
      }

      const botMessage: Message = {
        id: Date.now() + 1,
        text: botResponseText,
        sender: 'bot',
        timestamp: new Date()
      };

      setMessages(prev => [...prev, botMessage]);

    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: Message = {
        id: Date.now() + 1,
        text: 'Sorry, I encountered an error while processing your request. Please try again.',
        sender: 'bot',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const formatStockResponse = (data: ChatResponse): string => {
    let response = data.response || '';
    if (data.stock_data) {
      response += '\n\n📊 Detailed Information:';
      Object.entries(data.stock_data).forEach(([symbol, stockInfo]: [string, any]) => {
        response += `\n\n${symbol}:`;
        response += `\n💰 Price: $${stockInfo.price?.toFixed(2)}`;
        response += `\n📈 Change: ${stockInfo.change >= 0 ? '+' : ''}${stockInfo.change?.toFixed(2)} (${stockInfo.change_percent?.toFixed(2)}%)`;
        response += `\n📊 Volume: ${stockInfo.volume?.toLocaleString()}`;
        if (stockInfo.pe_ratio) response += `\n🏷️ P/E Ratio: ${stockInfo.pe_ratio?.toFixed(2)}`;
      });
    }
    return response;
  };

  const formatNewsResponse = (data: ChatResponse): string => {
    let response = data.response || '';
    if (data.news_items && data.news_items.length > 0) {
      response += '\n\n📰 Latest Headlines:';
      data.news_items.slice(0, 3).forEach((item, index) => {
        response += `\n\n${index + 1}. ${item.title}`;
        if (item.source) response += `\n   📰 Source: ${item.source}`;
        if (item.summary) response += `\n   📝 ${item.summary.substring(0, 100)}...`;
      });
    }
    return response;
  };

  const formatSentimentResponse = (data: ChatResponse): string => {
    let response = data.response || '';
    if (data.sentiment_data) {
      response += '\n\n📊 Sentiment Analysis:';
      Object.entries(data.sentiment_data).forEach(([key, sentiment]: [string, any]) => {
        const emoji = sentiment.sentiment === 'positive' ? '🟢' : 
                     sentiment.sentiment === 'negative' ? '🔴' : '🟡';
        response += `\n${emoji} ${key.toUpperCase()}: ${sentiment.sentiment} (${(sentiment.score * 100).toFixed(1)}%)`;
      });
    }
    return response;
  };

  const formatPortfolioResponse = (data: ChatResponse): string => {
    let response = data.response || '';
    if (data.portfolio) {
      response += '\n\n💼 Portfolio Summary:';
      response += `\n💰 Total Value: $${data.portfolio.total_value?.toFixed(2)}`;
      response += `\n📈 P&L: ${data.portfolio.total_profit_loss >= 0 ? '+' : ''}$${data.portfolio.total_profit_loss?.toFixed(2)}`;
      response += `\n📊 Performance: ${data.portfolio.total_profit_loss_percent?.toFixed(2)}%`;
    }
    return response;
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage(inputText);
    }
  };

  const formatTimestamp = (timestamp: Date): string => {
    return timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const suggestedQuestions = [
    "What's the current price of RELIANCE?",
    "Show me today's market news",
    "Analyze my portfolio",
    "What's the sentiment for TCS?",
    "Compare HDFC vs ICICI Bank"
  ];

  return (
    <div className="chat-window">
      <div className="messages-container">
        {messages.map((message) => (
          <div key={message.id} className={`message ${message.sender}`}>
            <div className="message-content">
              <div className="message-text">
                {message.text.split('\n').map((line, index) => (
                  <React.Fragment key={index}>
                    {line}
                    {index < message.text.split('\n').length - 1 && <br />}
                  </React.Fragment>
                ))}
              </div>
              <div className="message-time">
                {formatTimestamp(message.timestamp)}
              </div>
            </div>
          </div>
        ))}
        
        {isLoading && (
          <div className="message bot">
            <div className="message-content">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {messages.length === 1 && (
        <div className="suggested-questions">
          <p>Try asking me:</p>
          {suggestedQuestions.map((question, index) => (
            <button
              key={index}
              className="suggestion-button"
              onClick={() => sendMessage(question)}
            >
              {question}
            </button>
          ))}
        </div>
      )}

      <div className="input-container">
        <div className="input-wrapper">
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask about stocks, market news, portfolio analysis..."
            className="message-input"
            rows={1}
            disabled={isLoading}
          />
          <button
            onClick={() => sendMessage(inputText)}
            disabled={!inputText.trim() || isLoading}
            className="send-button"
          >
            {isLoading ? '⏳' : '📤'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatWindow;
