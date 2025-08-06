"""
CNBC Network 18 Scraper - Fetches news articles, market updates, and expert opinions
from CNBC TV18 (https://www.cnbctv18.com/)
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time
import random
import logging
import json
import re
from typing import Dict, List, Optional, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CnbcTv18Scraper:
    """Scraper for CNBC TV18 website"""
    
    def __init__(self, cache_duration: int = 3600):
        """Initialize CNBC TV18 scraper
        
        Args:
            cache_duration: Duration to cache results in seconds (default: 1 hour)
        """
        self.base_url = "https://www.cnbctv18.com"
        self.api_base_url = "https://www.cnbctv18.com/api/v1"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.cnbctv18.com/market/"
        }
        self.cache = {}
        self.cache_duration = cache_duration
        logger.info("CNBC TV18 scraper initialized")
        
    def _safe_get_link(self, link: Any) -> Optional[str]:
        """Safely get a string link from a potentially problematic value
        
        Args:
            link: Link that might be a string, list, or None
            
        Returns:
            Clean string link or None if not valid
        """
        # Handle None case
        if link is None:
            return None
            
        # Handle list case
        if isinstance(link, list):
            if not link:  # Empty list
                return None
            link = link[0]  # Take first item
            
        # Ensure it's a string
        if not isinstance(link, str):
            try:
                link = str(link)
            except:
                return None
                
        # Handle empty string
        if not link.strip():
            return None
            
        return link
        
    def _ensure_absolute_url(self, link: Any) -> Optional[str]:
        """Ensure a URL is absolute by adding base_url if needed
        
        Args:
            link: Link to process
            
        Returns:
            Absolute URL or None if not valid
        """
        link = self._safe_get_link(link)
        if not link:
            return None
            
        # Check if link is already absolute
        if isinstance(link, str) and not link.startswith("http"):
            return f"{self.base_url}{link if link.startswith('/') else '/' + link}"
        
        return link
        
    def _safe_extract_number(self, pattern: str, text: str, default: int = 1) -> int:
        """Safely extract a number from text using regex
        
        Args:
            pattern: Regex pattern with a capture group for the number
            text: Text to search in
            default: Default value to return if no match is found
            
        Returns:
            Extracted number or default value
        """
        if not text:
            return default
            
        try:
            match = re.search(pattern, text)
            if match and match.group(1):
                return int(match.group(1))
        except (ValueError, AttributeError, IndexError) as e:
            logger.debug(f"Error extracting number with pattern {pattern}: {str(e)}")
            
        return default
    
    def _make_request(self, url: str, params: Optional[Dict[str, Any]] = None, max_retries: int = 3) -> Optional[BeautifulSoup]:
        """Make an HTTP request and return BeautifulSoup object
        
        Args:
            url: URL to request
            params: Optional parameters for the request
            max_retries: Maximum number of retry attempts (default: 3)
            
        Returns:
            BeautifulSoup object or None if request failed
        """
        # List of user agents to rotate
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:95.0) Gecko/20100101 Firefox/95.0",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36"
        ]
        
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Rotate user agents and add a random delay to avoid being blocked
                headers = self.headers.copy()
                headers["User-Agent"] = random.choice(user_agents)
                
                # Random delay between requests (longer delay after each retry)
                delay = random.uniform(1, 3) * (retry_count + 1)
                time.sleep(delay)
                
                # Make request with timeout
                response = requests.get(url, headers=headers, params=params, timeout=10 + (5 * retry_count))
                
                # Handle different status codes
                if response.status_code == 200:
                    return BeautifulSoup(response.content, "html.parser")
                elif response.status_code in [403, 429]:  # Forbidden or Too Many Requests
                    logger.warning(f"Rate limited on {url}. Status: {response.status_code}. Retry: {retry_count + 1}/{max_retries}")
                    retry_count += 1
                    time.sleep(random.uniform(5, 10) * (retry_count + 1))  # Exponential backoff
                elif response.status_code in [404, 410]:  # Not found or gone
                    logger.warning(f"Page not available at {url}. Status: {response.status_code}")
                    return None
                else:
                    logger.warning(f"Failed to fetch {url}. Status code: {response.status_code}")
                    retry_count += 1
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout error fetching {url}. Retry: {retry_count + 1}/{max_retries}")
                retry_count += 1
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error fetching {url}. Retry: {retry_count + 1}/{max_retries}")
                retry_count += 1
            except Exception as e:
                logger.error(f"Error fetching {url}: {str(e)}")
                return None
                
        logger.error(f"Failed to fetch {url} after {max_retries} retries")
        return None
    
    def _make_api_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None, max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """Make an API request to CNBC TV18 API
        
        Args:
            endpoint: API endpoint
            params: Optional parameters for the request
            max_retries: Maximum number of retry attempts (default: 3)
            
        Returns:
            JSON response or None if request failed
        """
        url = f"{self.api_base_url}/{endpoint}"
        
        # List of user agents to rotate
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:95.0) Gecko/20100101 Firefox/95.0",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36"
        ]
        
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Rotate user agents and add a random delay to avoid being blocked
                headers = self.headers.copy()
                headers["User-Agent"] = random.choice(user_agents)
                headers["Accept"] = "application/json"
                
                # Random delay between requests (longer delay after each retry)
                delay = random.uniform(1, 3) * (retry_count + 1)
                time.sleep(delay)
                
                # Make request with timeout
                response = requests.get(url, headers=headers, params=params, timeout=10 + (5 * retry_count))
                
                # Handle different status codes
                if response.status_code == 200:
                    try:
                        return response.json()
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode JSON from {url}: {str(e)}")
                        retry_count += 1
                elif response.status_code in [403, 429]:  # Forbidden or Too Many Requests
                    logger.warning(f"Rate limited on API {url}. Status: {response.status_code}. Retry: {retry_count + 1}/{max_retries}")
                    retry_count += 1
                    time.sleep(random.uniform(5, 10) * (retry_count + 1))  # Exponential backoff
                elif response.status_code in [404, 410]:  # Not found or gone
                    logger.warning(f"API endpoint not available at {url}. Status: {response.status_code}")
                    return None
                else:
                    logger.warning(f"Failed to fetch API {url}. Status code: {response.status_code}")
                    retry_count += 1
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout error fetching API {url}. Retry: {retry_count + 1}/{max_retries}")
                retry_count += 1
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error fetching API {url}. Retry: {retry_count + 1}/{max_retries}")
                retry_count += 1
            except Exception as e:
                logger.error(f"Error fetching API {url}: {str(e)}")
                return None
                
        logger.error(f"Failed to fetch API {url} after {max_retries} retries")
        return None
    
    def get_top_news(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top financial news articles
        
        Args:
            limit: Maximum number of news articles to return
            
        Returns:
            List of news articles with title, summary, URL, and timestamp
        """
        cache_key = f"top_news_{limit}"
        if cache_key in self.cache:
            cache_time, cache_data = self.cache[cache_key]
            if (datetime.now() - cache_time).seconds < self.cache_duration:
                logger.info("Returning cached top news")
                return cache_data
        
        url = f"{self.base_url}/market/"
        soup = self._make_request(url)
        
        if not soup:
            return []
        
        news_articles = []
        
        try:
            # Find top stories section
            top_stories_section = soup.select(".top-stories-section")
            
            if top_stories_section:
                # Get articles from top stories
                articles = top_stories_section[0].select("article")[:limit]
                
                for article in articles:
                    try:
                        # Extract title and link
                        title_element = article.select_one("h2 a")
                        if not title_element:
                            continue
                        
                        title = title_element.text.strip()
                        link = self._ensure_absolute_url(title_element.get("href", ""))
                        
                        # Extract summary
                        summary_element = article.select_one(".article-list-text p")
                        summary = summary_element.text.strip() if summary_element else ""
                        
                        # Extract timestamp
                        time_element = article.select_one(".article-publish-time")
                        timestamp = datetime.now().isoformat()  # Default to current time
                        
                        if time_element:
                            timestamp_text = time_element.text.strip()
                            try:
                                # Handle relative time (e.g., "5 hours ago", "2 days ago")
                                if "min" in timestamp_text:
                                    minutes = self._safe_extract_number(r'(\d+)', timestamp_text)
                                    # Keep the default timestamp (current time)
                                elif "hour" in timestamp_text:
                                    hours = self._safe_extract_number(r'(\d+)', timestamp_text)
                                    # Approximate based on current time
                                    timestamp = datetime.now().replace(minute=0, second=0, microsecond=0).isoformat()
                                elif "day" in timestamp_text:
                                    days = self._safe_extract_number(r'(\d+)', timestamp_text)
                                    # Approximate based on current time
                                    timestamp = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
                                elif re.match(r'\d{2} [A-Za-z]{3} \d{4}', timestamp_text):  # Format like "01 Aug 2023"
                                    dt = datetime.strptime(timestamp_text, "%d %b %Y")
                                    timestamp = dt.isoformat()
                            except Exception as e:
                                logger.error(f"Error parsing timestamp: {e}")
                        
                        # Extract image URL
                        img_element = article.select_one("img")
                        image_url = img_element.get("src", "") if img_element else ""
                        
                        news_articles.append({
                            "title": title,
                            "summary": summary,
                            "url": link,
                            "source": "CNBC TV18",
                            "image_url": image_url,
                            "timestamp": timestamp,
                            "category": self._categorize_news(title, summary)
                        })
                    
                    except Exception as e:
                        logger.error(f"Error parsing news article: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error extracting top news: {e}")
        
        # Cache the results
        self.cache[cache_key] = (datetime.now(), news_articles)
        logger.info(f"Fetched {len(news_articles)} top news articles from CNBC TV18")
        
        return news_articles
    
    def get_expert_opinions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get expert opinions and market insights
        
        Args:
            limit: Maximum number of opinions to return
            
        Returns:
            List of expert opinions with analyst name, view, etc.
        """
        cache_key = f"expert_opinions_{limit}"
        if cache_key in self.cache:
            cache_time, cache_data = self.cache[cache_key]
            if (datetime.now() - cache_time).seconds < self.cache_duration:
                logger.info("Returning cached expert opinions")
                return cache_data
        
        url = f"{self.base_url}/market-expert-views/"
        soup = self._make_request(url)
        
        if not soup:
            return []
        
        expert_opinions = []
        
        try:
            # Find expert views
            articles = soup.select(".expert-views-section article")[:limit]
            
            for article in articles:
                try:
                    # Extract title and link
                    title_element = article.select_one("h2 a")
                    if not title_element:
                        continue
                    
                    title = title_element.text.strip()
                    link = title_element.get("href", "")
                    
                    # Ensure link is absolute
                    if link and not link.startswith("http"):
                        link = self.base_url + link
                    
                    # Extract expert name
                    expert_element = article.select_one(".expert-name")
                    expert_name = expert_element.text.strip() if expert_element else "Market Expert"
                    
                    # Extract company
                    company_element = article.select_one(".expert-company")
                    company = company_element.text.strip() if company_element else ""
                    
                    # Extract timestamp
                    time_element = article.select_one(".article-publish-time")
                    timestamp = datetime.now().isoformat()  # Default to current time
                    
                    if time_element:
                        try:
                            timestamp_text = time_element.text.strip()
                            if "min" in timestamp_text or "hour" in timestamp_text or "day" in timestamp_text:
                                # Keep default timestamp for relative times
                                pass
                            elif re.match(r'\d{2} [A-Za-z]{3} \d{4}', timestamp_text):
                                dt = datetime.strptime(timestamp_text, "%d %b %Y")
                                timestamp = dt.isoformat()
                        except:
                            pass
                    
                    # Visit the article page to get the summary/opinion
                    opinion_summary = ""
                    safe_link = self._safe_get_link(link)
                    if safe_link:
                        article_soup = self._make_request(safe_link)
                        if article_soup:
                            summary_elements = article_soup.select(".article-description p")
                            if summary_elements:
                                opinion_summary = " ".join([el.text.strip() for el in summary_elements[:2]])
                    
                    # Extract tickers mentioned
                    tickers = self._extract_tickers_from_text(title + " " + opinion_summary)
                    
                    expert_opinions.append({
                        "title": title,
                        "expert_name": expert_name,
                        "company": company,
                        "summary": opinion_summary[:300] + "..." if len(opinion_summary) > 300 else opinion_summary,
                        "url": link,
                        "tickers": tickers,
                        "timestamp": timestamp
                    })
                
                except Exception as e:
                    logger.error(f"Error parsing expert opinion: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error extracting expert opinions: {e}")
        
        # Cache the results
        self.cache[cache_key] = (datetime.now(), expert_opinions)
        logger.info(f"Fetched {len(expert_opinions)} expert opinions from CNBC TV18")
        
        return expert_opinions
    
    def get_market_alerts(self) -> List[Dict[str, Any]]:
        """Get latest market alerts and breaking news
        
        Returns:
            List of market alerts with timestamp and priority
        """
        cache_key = "market_alerts"
        if cache_key in self.cache:
            cache_time, cache_data = self.cache[cache_key]
            if (datetime.now() - cache_time).seconds < self.cache_duration / 4:  # Shorter cache for alerts
                logger.info("Returning cached market alerts")
                return cache_data
        
        url = f"{self.base_url}/market/"
        soup = self._make_request(url)
        
        if not soup:
            return []
        
        market_alerts = []
        
        try:
            # Find breaking news section
            breaking_section = soup.select(".breaking-news-section")
            
            if breaking_section:
                alert_items = breaking_section[0].select("li")
                
                for item in alert_items:
                    try:
                        text_element = item.select_one("a")
                        if not text_element:
                            continue
                        
                        text = text_element.text.strip()
                        link = text_element.get("href", "")
                        
                        # Ensure link is absolute
                        if link and not link.startswith("http"):
                            link = self.base_url + link
                        
                        # Check if it's labeled as breaking news
                        is_breaking = "breaking" in item.get("class", [])
                        priority = "high" if is_breaking else "medium"
                        
                        market_alerts.append({
                            "alert_text": text,
                            "url": link,
                            "priority": priority,
                            "timestamp": datetime.now().isoformat(),
                            "tickers": self._extract_tickers_from_text(text)
                        })
                    
                    except Exception as e:
                        logger.error(f"Error parsing market alert: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error extracting market alerts: {e}")
        
        # Cache the results
        self.cache[cache_key] = (datetime.now(), market_alerts)
        logger.info(f"Fetched {len(market_alerts)} market alerts from CNBC TV18")
        
        return market_alerts
    
    def get_sector_news(self, sector: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get news for a specific sector
        
        Args:
            sector: Sector name (e.g., "banking", "auto", "technology")
            limit: Maximum number of news articles to return
            
        Returns:
            List of news articles for the sector
        """
        cache_key = f"sector_news_{sector}_{limit}"
        if cache_key in self.cache:
            cache_time, cache_data = self.cache[cache_key]
            if (datetime.now() - cache_time).seconds < self.cache_duration:
                logger.info(f"Returning cached news for {sector} sector")
                return cache_data
        
        # Map sector to URL segment
        sector_map = {
            "banking": "finance",
            "auto": "auto",
            "technology": "technology",
            "pharma": "healthcare",
            "energy": "energy",
            "telecom": "telecom",
            "consumer": "retail",
            "metal": "infrastructure",
            "realty": "realty"
        }
        
        sector_url = sector_map.get(sector.lower(), sector.lower())
        url = f"{self.base_url}/{sector_url}/"
        soup = self._make_request(url)
        
        if not soup:
            return []
        
        sector_news = []
        
        try:
            # Find articles in the sector page
            articles = soup.select(".sector-news-list article")[:limit]
            
            for article in articles:
                try:
                    # Extract title and link
                    title_element = article.select_one("h2 a")
                    if not title_element:
                        continue
                    
                    title = title_element.text.strip()
                    link = title_element.get("href", "")
                    
                    # Ensure link is absolute
                    if link and not link.startswith("http"):
                        link = self.base_url + link
                    
                    # Extract summary
                    summary_element = article.select_one(".article-list-text p")
                    summary = summary_element.text.strip() if summary_element else ""
                    
                    # Extract timestamp
                    time_element = article.select_one(".article-publish-time")
                    timestamp = datetime.now().isoformat()
                    
                    if time_element:
                        timestamp_text = time_element.text.strip()
                        try:
                            # Handle relative time or formatted date
                            if re.match(r'\d{2} [A-Za-z]{3} \d{4}', timestamp_text):
                                dt = datetime.strptime(timestamp_text, "%d %b %Y")
                                timestamp = dt.isoformat()
                        except:
                            pass  # Keep default timestamp
                    
                    sector_news.append({
                        "title": title,
                        "summary": summary,
                        "url": link,
                        "source": "CNBC TV18",
                        "sector": sector,
                        "timestamp": timestamp,
                        "tickers": self._extract_tickers_from_text(title + " " + summary)
                    })
                
                except Exception as e:
                    logger.error(f"Error parsing sector news: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error extracting sector news: {e}")
        
        # Cache the results
        self.cache[cache_key] = (datetime.now(), sector_news)
        logger.info(f"Fetched {len(sector_news)} news articles for {sector} sector from CNBC TV18")
        
        return sector_news
    
    def clear_cache(self):
        """Clear the cache"""
        self.cache = {}
        logger.info("CNBC TV18 scraper cache cleared")
    
    def _categorize_news(self, title: str, content: str) -> str:
        """Categorize news based on content
        
        Args:
            title: News title
            content: News content
            
        Returns:
            News category
        """
        title_content = (title + " " + content).lower()
        
        if any(term in title_content for term in ["q1", "q2", "q3", "q4", "quarter", "earnings", "profit", "revenue", "financials"]):
            return "earnings"
        elif any(term in title_content for term in ["rbi", "sebi", "policy", "regulation", "government", "ministry"]):
            return "regulatory"
        elif any(term in title_content for term in ["buy", "sell", "invest", "exit", "accumulate", "hold"]):
            return "recommendation"
        elif any(term in title_content for term in ["merger", "acquisition", "takeover", "bid", "deal"]):
            return "corporate_action"
        elif any(term in title_content for term in ["ipo", "public issue", "listing", "debut"]):
            return "ipo"
        elif any(term in title_content for term in ["stock", "share", "market", "nifty", "sensex"]):
            return "market_news"
        else:
            return "general"
    
    def _extract_tickers_from_text(self, text: str) -> List[str]:
        """Extract stock tickers from text
        
        Args:
            text: Text to extract tickers from
            
        Returns:
            List of extracted tickers
        """
        # Common Indian stock tickers
        common_tickers = [
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR", 
            "ICICIBANK", "KOTAKBANK", "SBIN", "BHARTIARTL", "ASIANPAINT",
            "BAJFINANCE", "AXISBANK", "LT", "WIPRO", "MARUTI", "HCLTECH",
            "TATASTEEL", "ITC", "TITAN", "SUNPHARMA", "ULTRACEMCO", "TECHM",
            "ONGC", "BAJAJ-AUTO", "ADANIPORTS"
        ]
        
        found_tickers = []
        
        for ticker in common_tickers:
            if re.search(r'\b' + ticker + r'\b', text.upper()):
                found_tickers.append(ticker)
        
        return found_tickers

# Create singleton instance
cnbc_tv18_scraper = CnbcTv18Scraper()
