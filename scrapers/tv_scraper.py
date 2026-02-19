#!/usr/bin/env python3
"""
TradingView Community Indicator Scraper
Fetches Pine Script indicators from TradingView's community scripts
"""

import requests
import json
import time
import os
from datetime import datetime

# TradingView API endpoints
BASE_URL = "https://scanner.tradingview.com"
SCANNER_ENDPOINT = "/scanner/scan"

# Categories to scrape
CATEGORIES = [
    {"name": "Editors Picks", "sort": "popular", "desc": ""},
    {"name": "Top Rising", "sort": "trend", "desc": ""},
    {"name": "Most Popular", "sort": "popular", "desc": "all_time"},
]

# Fields to request
FIELDS = [
    "name",
    "description",
    "short_description",
    "symbol",
    "type",
    "id",
    "pубликация",
    "author_name",
    "author_uri",
    "created",
    "updated",
    "closed",
    "price",
    "access",
    "payment",
    "pro_status",
    "indicate",
    "scripts_count",
    "like_count",
    "dislike_count",
    "comment_count",
    "relikes",
    "recomments",
    "uricode",
    "dtype",
]

def get_indicators(category="Editors Picks", limit=100):
    """
    Fetch indicators from TradingView scanner
    
    Args:
        category: Category name (Editors Picks, Top Rising, etc.)
        limit: Number of results to fetch
    
    Returns:
        List of indicator dictionaries
    """
    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    
    # Construct scan request for Pine indicators
    data = {
        "columns": FIELDS,
        "filter": [
            {"left": "type", "operation": "equal", "right": "script|pine"},
            {"left": "access", "operation": "equal", "right": "public"},
        ],
        "sort": {"sortBy": "like_count", "sortOrder": "desc"},
        "range": {"offset": 0, "limit": limit},
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}{SCANNER_ENDPOINT}",
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        
        indicators = result.get("data", [])
        print(f"Found {len(indicators)} indicators in {category}")
        return indicators
        
    except requests.RequestException as e:
        print(f"Error fetching indicators: {e}")
        return []

def get_pine_script_code(script_id):
    """
    Get the Pine Script source code for a specific indicator
    
    Args:
        script_id: The TradingView script ID
    
    Returns:
        Pine Script source code as string
    """
    # This would require actual TradingView API authentication
    # For now, we'll use a placeholder that converts to the full flow
    # In production, you'd need proper TV API access
    return None

def save_indicators(indicators, output_file="data/indicators.json"):
    """Save indicators to JSON file"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(indicators, f, indent=2, default=str)
    
    print(f"Saved {len(indicators)} indicators to {output_file}")

def main():
    """Main scraping function"""
    print("=" * 50)
    print("TradingView Indicator Scraper")
    print("=" * 50)
    
    all_indicators = []
    
    for category in CATEGORIES:
        print(f"\nFetching {category['name']}...")
        indicators = get_indicators(category["name"])
        all_indicators.extend(indicators)
        time.sleep(1)  # Rate limiting
    
    # Remove duplicates by ID
    unique_indicators = {ind["id"]: ind for ind in all_indicators}.values()
    
    print(f"\nTotal unique indicators: {len(unique_indicators)}")
    save_indicators(list(unique_indicators))
    
    return list(unique_indicators)

if __name__ == "__main__":
    main()
