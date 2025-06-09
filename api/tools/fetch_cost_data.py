"""
MCP Tool: Fetch Cost Data

Tool for fetching real-time cost of living data from Numbeo using BrightData.
This tool can fetch cost index data for any city on demand.
"""

import asyncio
import logging
from typing import Dict, Any
from pydantic import BaseModel, Field
import sys
import os

# Add datasource_integrations to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../datasource_integrations'))

try:
    from numbeo_cost_index import fetch_cost_index_for_city
except ImportError:
    # Fallback if import fails
    async def fetch_cost_index_for_city(city: str) -> Dict:
        raise NotImplementedError("Numbeo cost index fetcher not available")

logger = logging.getLogger(__name__)

class FetchCostDataArgs(BaseModel):
    """Arguments for fetching cost data"""
    city: str = Field(
        description="Name of the city to fetch cost data for (e.g., 'Lisbon', 'Berlin')"
    )

async def fetch_cost_data(args: FetchCostDataArgs) -> Dict[str, Any]:
    """
    Fetch real-time cost of living data for a specific city from Numbeo.
    
    This tool uses BrightData to scrape Numbeo's cost-of-living pages,
    bypassing anti-bot protections to get accurate, up-to-date cost indices.
    
    Args:
        args: FetchCostDataArgs containing the city name
        
    Returns:
        Dict containing city name, cost index, and timestamp
        
    Raises:
        ValueError: If city parameter is missing
        RuntimeError: If BrightData authentication is not configured
        Exception: If fetching fails
    """
    try:
        logger.info(f"Fetching cost data for {args.city}")
        
        # Validate environment
        endpoint = os.getenv("BRIGHTDATA_ENDPOINT")
        if not endpoint or "brd.superproxy.io" not in endpoint:
            raise RuntimeError(
                "BrightData endpoint not configured. "
                "Set BRIGHTDATA_ENDPOINT environment variable."
            )
        
        # Fetch cost data using BrightData
        result = await fetch_cost_index_for_city(args.city)
        
        logger.info(f"Successfully fetched cost data for {args.city}: {result['cost_index']}")
        
        return {
            "success": True,
            "city": result["city"],
            "cost_index": result["cost_index"],
            "last_updated": result["last_updated"],
            "description": f"Cost index for {result['city']} is {result['cost_index']} (higher = more expensive)"
        }
        
    except Exception as e:
        logger.error(f"Error fetching cost data for {args.city}: {e}")
        return {
            "success": False,
            "error": str(e),
            "city": args.city,
            "description": f"Failed to fetch cost data for {args.city}: {str(e)}"
        }

# Tool metadata for MCP registration
TOOL_METADATA = {
    "name": "fetch_cost_data",
    "description": "Fetch real-time cost of living data for any city from Numbeo using BrightData",
    "parameters": FetchCostDataArgs.model_json_schema(),
    "function": fetch_cost_data
} 