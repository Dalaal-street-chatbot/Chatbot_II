// API Configuration
// This file centralizes all API endpoint configurations

// Base API URL - change this when deploying to production
const getBaseUrl = (): string => {
  // For production deployment, you'll need to set your actual backend URL
  if (process.env.NODE_ENV === 'production') {
    // Replace with your actual Azure backend URL when you deploy the backend
    return process.env.REACT_APP_API_BASE_URL || 'https://your-backend-app-name.azurewebsites.net';
  }
  // For local development
  return 'http://localhost:8000';
};

export const API_BASE_URL = getBaseUrl();

// API Endpoints
export const API_ENDPOINTS = {
  // Main chat endpoints
  CHAT: `${API_BASE_URL}/api/v1/chat`,
  GOOGLE_CLOUD_CHAT: `${API_BASE_URL}/api/v1/google-cloud/financial-chat`,
  
  // Stock and market data
  STOCK: `${API_BASE_URL}/api/v1/stock`,
  INDICES: `${API_BASE_URL}/api/v1/indices`,
  CHART_DATA: `${API_BASE_URL}/api/v1/chart-data`,
  
  // News and analysis
  NEWS: `${API_BASE_URL}/api/v1/news`,
  ANALYSIS: `${API_BASE_URL}/api/v1/analysis`,
} as const;

// Helper function to build API URLs with query parameters
export const buildApiUrl = (endpoint: string, params?: Record<string, string | number>): string => {
  if (!params) return endpoint;
  
  const url = new URL(endpoint);
  Object.entries(params).forEach(([key, value]) => {
    url.searchParams.append(key, String(value));
  });
  
  return url.toString();
};

// API request headers
export const getApiHeaders = (): HeadersInit => ({
  'Content-Type': 'application/json',
  'Accept': 'application/json',
});

// Export for use in components
export default {
  API_BASE_URL,
  API_ENDPOINTS,
  buildApiUrl,
  getApiHeaders,
};
