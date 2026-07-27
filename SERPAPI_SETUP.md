# SerpAPI Setup for Sector News Feature

This application uses SerpAPI to fetch recent news articles about industrial sectors.

## Setup Instructions

1. **Get a SerpAPI Key:**
   - Sign up at https://serpapi.com/
   - Get your free API key from the dashboard (100 free searches per month)

2. **Set Environment Variable:**
   
   **Windows (PowerShell):**
   ```powershell
   $env:SERPAPI_KEY="your_api_key_here"
   ```
   
   **Windows (Command Prompt):**
   ```cmd
   set SERPAPI_KEY=your_api_key_here
   ```
   
   **Linux/Mac:**
   ```bash
   export SERPAPI_KEY="your_api_key_here"
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install SerpAPI directly:
   ```bash
   pip install google-search-results
   ```

4. **Run the Application:**
   ```bash
   uvicorn api:api --reload
   ```

## How It Works

- When you search for an industrial code, the system automatically fetches the latest news (last 24 hours) related to that sector
- The news appears in a dedicated section below the code results
- If SerpAPI is not configured or fails, the feature gracefully degrades (no error shown, just no news section)

## Notes

- The feature works without SerpAPI (it just won't show news)
- News is fetched for the top matching industrial code
- Results are limited to the past 24 hours and top 5 articles

