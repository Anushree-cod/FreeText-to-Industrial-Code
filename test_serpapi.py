"""
Test script to check SerpAPI setup
Run this to diagnose why news search isn't working
"""

import os

print("=" * 50)
print("SerpAPI Setup Diagnostic")
print("=" * 50)

# Check 1: Package installation
print("\n1. Checking if SerpAPI package is installed...")
try:
    from serpapi import GoogleSearch
    print("   ✓ Package 'serpapi' found")
    package_ok = True
except ImportError:
    try:
        from google_search_results import GoogleSearch
        print("   ✓ Package 'google-search-results' found")
        package_ok = True
    except ImportError:
        print("   ✗ Package NOT installed")
        print("   → Install with: pip install google-search-results")
        package_ok = False

# Check 2: API Key (check both direct setting and environment variable)
print("\n2. Checking for API key...")
try:
    # Try to import from api.py to check direct setting
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from api import SERPAPI_KEY_DIRECT
    if SERPAPI_KEY_DIRECT:
        api_key = SERPAPI_KEY_DIRECT
        print(f"   ✓ API Key found in code (SERPAPI_KEY_DIRECT): {api_key[:10]}...{api_key[-4:]}")
        key_ok = True
    else:
        api_key = os.getenv("SERPAPI_KEY")
        if api_key:
            print(f"   ✓ API Key found in environment: {api_key[:10]}...{api_key[-4:]}")
            key_ok = True
        else:
            print("   ✗ API Key NOT set")
            print("   → Option 1: Set SERPAPI_KEY_DIRECT in api.py (line ~30)")
            print("   → Option 2: Set environment variable: $env:SERPAPI_KEY='your_api_key_here'")
            print("   → Get your key from: https://serpapi.com/dashboard")
            key_ok = False
except Exception as e:
    api_key = os.getenv("SERPAPI_KEY")
    if api_key:
        print(f"   ✓ API Key found in environment: {api_key[:10]}...{api_key[-4:]}")
        key_ok = True
    else:
        print("   ✗ API Key NOT set")
        print("   → Option 1: Set SERPAPI_KEY_DIRECT in api.py (line ~30)")
        print("   → Option 2: Set environment variable: $env:SERPAPI_KEY='your_api_key_here'")
        print("   → Get your key from: https://serpapi.com/dashboard")
        key_ok = False

# Check 3: Test API call (if both are OK)
if package_ok and key_ok:
    print("\n3. Testing API call...")
    try:
        params = {
            "engine": "google",
            "q": "manufacturing industry news",
            "tbm": "nws",
            "api_key": api_key,
            "num": 3,
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        
        if "error" in results:
            print(f"   ✗ API Error: {results.get('error')}")
            print(f"   → Check your API key and account status")
        elif "news_results" in results:
            news_count = len(results.get("news_results", []))
            print(f"   ✓ API call successful! Found {news_count} news items")
            if news_count > 0:
                print(f"   → First result: {results['news_results'][0].get('title', 'N/A')[:50]}...")
        else:
            print(f"   ⚠ API call completed but no news_results found")
            print(f"   → Response keys: {list(results.keys())}")
    except Exception as e:
        print(f"   ✗ API call failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n3. Skipping API test (fix issues above first)")

print("\n" + "=" * 50)
if package_ok and key_ok:
    print("Setup looks good! If news still doesn't show, check server logs.")
else:
    print("Please fix the issues above and run this script again.")
print("=" * 50)

