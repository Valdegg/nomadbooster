"""
Airbnb Accommodation Cost Lookup Tool

Tool for fetching accommodation costs for a specific city on demand.
Wraps the core Airbnb scraping functionality in a simple tool interface.

Tool Usage: {"city": "Berlin", "property_type": "both"}
Returns: {"city": "Berlin", "entire_place_eur": 316, "private_room_eur": 80}
"""

import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Literal
import os
from pydantic import BaseModel, Field

# Import core functions from the main airbnb module
from airbnb_accommodation_costs import fetch_airbnb_data_for_city

logger = logging.getLogger(__name__)


class AccommodationLookupArgs(BaseModel):
    """
    Arguments schema for Airbnb accommodation cost lookup tool
    """
    city: str = Field(
        description="City name to search for accommodation (e.g., 'Berlin', 'Barcelona')",
        examples=["Berlin", "Barcelona", "Amsterdam", "Prague"]
    )
    
    property_type: Literal["entire_place", "private_room", "both"] = Field(
        default="both",
        description="Type of accommodation to search for"
    )
    
    checkin_days_ahead: int = Field(
        default=7,
        ge=1,
        le=365,
        description="Number of days ahead for check-in date (1-365)"
    )
    
    stay_duration_days: int = Field(
        default=3,
        ge=1,
        le=30,
        description="Length of stay in days (1-30)"
    )
    
    guests: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Number of guests (1-8)"
    )

async def tool_fetch_accommodation_cost(args: dict) -> dict:
    """
    Tool for fetching Airbnb accommodation costs for a specific city
    
    Args:
        args: Dictionary that will be validated against AccommodationLookupArgs schema
        
    Returns:
        dict: {
            "city": "Berlin",
            "entire_place_eur": 316,             # Median price for entire places (if requested)
            "private_room_eur": 80,              # Median price for private rooms (if requested)
            "entire_place_range": {"min": 72, "max": 208, "sample_size": 20},
            "private_room_range": {"min": 42, "max": 114, "sample_size": 20},
            "travel_dates": "2025-06-15 to 2025-06-18",
            "guests": 1,
            "last_updated": "2025-06-08T12:06:39.754735"
        }
    """
    
    # Validate and parse arguments using Pydantic
    try:
        validated_args = AccommodationLookupArgs(**args)
    except Exception as e:
        raise ValueError(f"Invalid arguments: {e}")
    
    # Extract validated values
    city = validated_args.city
    property_type = validated_args.property_type
    checkin_days_ahead = validated_args.checkin_days_ahead
    stay_duration_days = validated_args.stay_duration_days
    guests = validated_args.guests
    
    # Calculate dates
    checkin_date = datetime.now() + timedelta(days=checkin_days_ahead)
    checkout_date = checkin_date + timedelta(days=stay_duration_days)
    
    logger.info(f"Fetching accommodation costs for {city} ({checkin_date.date()} to {checkout_date.date()}, {guests} guest(s))")
    
    result = {
        "city": city,
        "travel_dates": f"{checkin_date.date()} to {checkout_date.date()}",
        "guests": guests,
        "stay_duration_days": stay_duration_days,
        "last_updated": datetime.now().isoformat()
    }
    
    # Determine which property types to fetch
    types_to_fetch = []
    if property_type in ["entire_place", "both"]:
        types_to_fetch.append("entire_place")
    if property_type in ["private_room", "both"]:
        types_to_fetch.append("private_room")
    
    # Fetch data for each property type
    for prop_type in types_to_fetch:
        try:
            logger.info(f"Fetching {prop_type} data for {city}...")
            
            search_data = await fetch_airbnb_data_for_city(
                city=city,
                checkin_date=checkin_date,
                checkout_date=checkout_date,
                guests=guests,
                property_type=prop_type
            )
            
            # Extract key metrics
            if prop_type == "entire_place":
                result["entire_place_eur"] = search_data["accommodation_cost_eur"]
                result["entire_place_range"] = {
                    "min": search_data["price_range"]["min"],
                    "max": search_data["price_range"]["max"],
                    "sample_size": search_data["sample_size"]
                }
            else:  # private_room
                result["private_room_eur"] = search_data["accommodation_cost_eur"]
                result["private_room_range"] = {
                    "min": search_data["price_range"]["min"],
                    "max": search_data["price_range"]["max"],
                    "sample_size": search_data["sample_size"]
                }
            
            logger.info(f"✅ {prop_type} for {city}: €{search_data['accommodation_cost_eur']} median")
            
            # Small delay between property types
            if len(types_to_fetch) > 1 and prop_type != types_to_fetch[-1]:
                await asyncio.sleep(5)
                
        except Exception as e:
            logger.error(f"Failed to fetch {prop_type} data for {city}: {e}")
            # Set error values but continue
            if prop_type == "entire_place":
                result["entire_place_eur"] = None
                result["entire_place_error"] = str(e)
            else:
                result["private_room_eur"] = None
                result["private_room_error"] = str(e)
    
    return result

def get_tool_schema() -> dict:
    """
    Get the JSON schema for the tool arguments
    
    Returns:
        dict: JSON schema that can be used by MCP or other tool systems
    """
    return AccommodationLookupArgs.model_json_schema()

def main():
    """
    Test the tool with a sample city
    """
    try:
        logging.basicConfig(level=logging.INFO)
        
        # Show the tool schema
        print("=== Tool Schema ===")
        schema = get_tool_schema()
        print(json.dumps(schema, indent=2))
        
        # Test the tool
        test_args = {
            "city": "Berlin",
            "property_type": "both"
        }
        
        logger.info("Testing accommodation cost lookup tool...")
        result = asyncio.run(tool_fetch_accommodation_cost(test_args))
        
        print("\n=== Tool Result ===")
        print(json.dumps(result, indent=2))
        
        return result
        
    except Exception as e:
        logger.error(f"Tool test failed: {e}")
        raise

if __name__ == "__main__":
    main() 