# Bright Data MCP Server Setup

This directory contains scripts and tools for setting up and testing the Bright Data MCP (Model Context Protocol) server integration.

## 🚀 Quick Start

1. **Setup the MCP Server**:
   ```bash
   cd mcp
   chmod +x setup_brightdata_mcp.sh
   ./setup_brightdata_mcp.sh
   ```

2. **Set your API Token**:
   ```bash
   # Option 1: Environment variable
   export BRIGHTDATA_API_TOKEN=your_token_here
   
   # Option 2: Create .env file
   echo "BRIGHTDATA_API_TOKEN=your_token_here" > .env
   
   # Optional: Set browser zone (replaces deprecated BROWSER_AUTH)
   export BRIGHTDATA_BROWSER_ZONE=your_browser_zone_name
   ```

3. **Run the MCP Server**:
   ```bash
   chmod +x run_mcp_server.sh
   ./run_mcp_server.sh
   ```

4. **Install MCP Client**:
   ```bash
   chmod +x install_mcp_client.sh
   ./install_mcp_client.sh
   ```

5. **Test MCP Tools**:
   ```bash
   python mcp_client_test.py
   ```

## 📁 Files Overview

- **`setup_brightdata_mcp.sh`**: Installation and setup script
- **`run_mcp_server.sh`**: Start the MCP server
- **`install_mcp_client.sh`**: Install MCP client dependencies
- **`mcp_client_test.py`**: Real MCP client test with actual tool calls
- **`test_mcp_tools.py`**: Mock test framework (placeholder)
- **`setup_env.sh`**: Helper script to create .env file
- **`requirements.txt`**: Python dependencies for MCP client
- **`.env`**: Environment variables (created automatically)

## 🛠️ Available Tools

The Bright Data MCP server provides these tools:

### Web Scraping Tools
- `scrape_as_markdown` - Scrape webpage as markdown
- `scrape_as_html` - Scrape webpage as HTML  
- `search_engine` - Search Google, Bing, etc.

### Browser Automation Tools
- `scraping_browser_navigate` - Navigate to URLs
- `scraping_browser_click` - Click elements
- `scraping_browser_type` - Type text
- `scraping_browser_get_text` - Extract text
- `scraping_browser_screenshot` - Take screenshots

### Structured Data Tools
- `web_data_amazon_product` - Amazon product data
- `web_data_linkedin_profile` - LinkedIn profiles
- `web_data_instagram_post` - Instagram posts
- `web_data_zillow_listing` - Real estate data

## 🎯 Skyscanner Integration Strategy

Our planned approach for getting flight data:

1. **Browser Automation Workflow**:
   - Navigate to Skyscanner
   - Fill search form (origin, destination, dates)
   - Submit search
   - Extract flight results

2. **Tools Used**:
   - `scraping_browser_navigate` → Go to Skyscanner
   - `scraping_browser_click` → Interact with form fields
   - `scraping_browser_type` → Enter search criteria
   - `scraping_browser_get_text` → Extract flight data

3. **Benefits**:
   - Real browser simulation
   - Bypass bot detection
   - Handle JavaScript-heavy pages
   - Bright Data proxy network

## 🔧 Prerequisites

- Node.js and npm installed
- Bright Data API token
- Python 3.7+ (for testing scripts)

## 📝 Get API Token

1. Go to [Bright Data Control Panel](https://brightdata.com/cp/zones)
2. Create a new zone or use existing one
3. Copy the API token
4. Set it in environment or .env file

## 🚀 Next Steps

1. **Setup MCP Server**: Run the setup script
2. **Test Connection**: Verify tools are available
3. **Implement Integration**: Connect MCP tools to travel agent
4. **Test Flight Searches**: Real Skyscanner data extraction

## 🔗 Integration with Travel Agent

The MCP tools will be integrated into our travel recommendation system:

- **Flight Cost Tool**: Use MCP to get real-time flight prices
- **Accommodation Tool**: Already working with BrightData directly
- **Combined Recommendations**: Mix real flight costs with accommodation and climate data

Perfect for the **Bright Data MCP Hackathon**! 🏆 