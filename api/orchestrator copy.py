import json
import os
from typing import Dict, List, Any, AsyncGenerator, Optional
import pandas as pd
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel as LangChainBaseModel, Field as LangChainField, validator
import logging
import asyncio
from datetime import datetime

# Import accommodation lookup tool  
from tools.tool_airbnb_accommodation_lookup import tool_fetch_accommodation_cost

# Import flight lookup tool
from tools.dohop_flight_lookup import lookup_dohop_flights
from datetime import datetime

logger = logging.getLogger(__name__)

# Flight data persistence
FLIGHT_DATA_FILE = "../data/flight_results.json"

def save_flight_data(flight_result: dict, origin: str, destination: str):
    """Save flight data to persistent JSON file"""
    try:
        # Load existing data
        if os.path.exists(FLIGHT_DATA_FILE):
            with open(FLIGHT_DATA_FILE, 'r') as f:
                flight_data = json.load(f)
        else:
            flight_data = {}
        
        # Create key for this route
        route_key = f"{origin}-{destination}"
        
        # Add timestamp and save
        flight_result_with_timestamp = flight_result.copy()
        flight_result_with_timestamp['saved_at'] = datetime.now().isoformat()
        flight_result_with_timestamp['route_key'] = route_key
        
        flight_data[route_key] = flight_result_with_timestamp
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(FLIGHT_DATA_FILE), exist_ok=True)
        
        # Save to file
        with open(FLIGHT_DATA_FILE, 'w') as f:
            json.dump(flight_data, f, indent=2)
            
        logger.info(f"Saved flight data for route {route_key}")
        
    except Exception as e:
        logger.error(f"Error saving flight data: {e}")

def load_flight_data() -> dict:
    """Load all saved flight data"""
    try:
        if os.path.exists(FLIGHT_DATA_FILE):
            with open(FLIGHT_DATA_FILE, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"Error loading flight data: {e}")
        return {}

def get_flight_data_for_city(city_name: str, flight_data: dict = None) -> dict:
    """Get flight data for a specific city (try common airport codes)"""
    if flight_data is None:
        flight_data = load_flight_data()
    
    # Common airport codes for European cities
    city_airport_map = {
        'Berlin': ['BER', 'TXL', 'SXF'],
        'Paris': ['CDG', 'ORY', 'PAR'],
        'London': ['LHR', 'LGW', 'STN', 'LON'],
        'Madrid': ['MAD'],
        'Rome': ['FCO', 'CIA', 'ROM'],
        'Barcelona': ['BCN'],
        'Amsterdam': ['AMS'],
        'Munich': ['MUC'],
        'Vienna': ['VIE'],
        'Zurich': ['ZUR'],
        'Prague': ['PRG'],
        'Budapest': ['BUD'],
        'Warsaw': ['WAW'],
        'Stockholm': ['ARN', 'STO'],
        'Copenhagen': ['CPH'],
        'Oslo': ['OSL'],
        'Helsinki': ['HEL'],
        'Brussels': ['BRU'],
        'Lisbon': ['LIS'],
        'Dublin': ['DUB'],
        'Reykjavik': ['KEF', 'RKV']
    }
    
    # Get possible airport codes for this city
    airport_codes = city_airport_map.get(city_name, [city_name[:3].upper()])
    
    # Find flight data for any route ending at this city
    city_flight_data = {}
    for route_key, data in flight_data.items():
        origin, destination = route_key.split('-')
        if destination in airport_codes:
            # Use origin as key to avoid conflicts
            city_flight_data[origin] = data
    
    return city_flight_data

# Global state for search space tracking
class SearchState:
    def __init__(self):
        self.all_cities = pd.read_csv("../data/cities_static_properties_with_accommodation.csv")
        self.current_cities = self.all_cities.copy()
        self.applied_filters = {}
        
    def apply_filter(self, filter_name: str, filter_params: dict, filtered_df: pd.DataFrame):
        """Apply a filter and update state"""
        # Check if this is a replacement of the same filter type
        is_replacement = filter_name in self.applied_filters
        
        # Store the new filter
        self.applied_filters[filter_name] = filter_params
        
        if is_replacement:
            # Same filter type being replaced - reapply all filters from scratch
            # to avoid filtering on top of already filtered results
            self._reapply_all_filters()
        else:
            # Different filter type being added - stack on current results
            self._apply_single_filter(filter_name, filter_params)
    
    def _apply_single_filter(self, filter_name: str, filter_params: dict):
        """Apply a single filter to current cities"""
        if filter_name == "climate":
            self._apply_climate_filter(filter_params)
        elif filter_name == "budget":
            self._apply_budget_filter(filter_params)
        elif filter_name == "safety":
            self._apply_safety_filter(filter_params)
        elif filter_name == "language":
            self._apply_language_filter(filter_params)
        elif filter_name == "healthcare":
            self._apply_healthcare_filter(filter_params)
        elif filter_name == "pollution":
            self._apply_pollution_filter(filter_params)
        elif filter_name == "public_transport":
            self._apply_public_transport_filter(filter_params)
        elif filter_name == "urban_nature":
            self._apply_urban_nature_filter(filter_params)
        elif filter_name == "city_size":
            self._apply_city_size_filter(filter_params)
        elif filter_name == "walkability":
            self._apply_walkability_filter(filter_params)
        elif filter_name == "item_costs":
            self._apply_item_costs_filter(filter_params)
    
    def _reapply_all_filters(self):
        """Reapply all stored filters from scratch to avoid stacking same filter types"""
        # Start with all cities
        self.current_cities = self.all_cities.copy()
        
        # Apply each filter in order
        for filter_name, filter_params in self.applied_filters.items():
            if filter_name == "climate":
                self._apply_climate_filter(filter_params)
            elif filter_name == "budget":
                self._apply_budget_filter(filter_params)
            elif filter_name == "safety":
                self._apply_safety_filter(filter_params)
            elif filter_name == "language":
                self._apply_language_filter(filter_params)
            elif filter_name == "healthcare":
                self._apply_healthcare_filter(filter_params)
            elif filter_name == "pollution":
                self._apply_pollution_filter(filter_params)
            elif filter_name == "public_transport":
                self._apply_public_transport_filter(filter_params)
            elif filter_name == "urban_nature":
                self._apply_urban_nature_filter(filter_params)
            elif filter_name == "city_size":
                self._apply_city_size_filter(filter_params)
            elif filter_name == "walkability":
                self._apply_walkability_filter(filter_params)
            elif filter_name == "item_costs":
                self._apply_item_costs_filter(filter_params)
    
    def _apply_climate_filter(self, filter_params):
        """Apply climate filter logic using comprehensive weather data"""
        min_temp = filter_params.get("min_temp")
        max_temp = filter_params.get("max_temp")
        max_rainfall = filter_params.get("max_rainfall")
        travel_month = filter_params.get("travel_month")
        min_sunshine_score = filter_params.get("min_sunshine_score")
        max_uv_index = filter_params.get("max_uv_index")
        max_precip_probability = filter_params.get("max_precip_probability")
        sunshine_category = filter_params.get("sunshine_category")
        rain_category = filter_params.get("rain_category")
        
        try:
            # Load comprehensive weather data
            weather_data = pd.read_csv("../data/cities_weather.csv")
            filtered_weather_data = weather_data.copy()
            
            # Filter by travel month first if specified
            if travel_month is not None:
                filtered_weather_data = filtered_weather_data[filtered_weather_data['travel_month'] == travel_month]
            
            # Apply temperature filters (use temp_max_c and temp_min_c for more precise filtering)
            if min_temp is not None:
                filtered_weather_data = filtered_weather_data[filtered_weather_data['temp_min_c'] >= min_temp]
            if max_temp is not None:
                filtered_weather_data = filtered_weather_data[filtered_weather_data['temp_max_c'] <= max_temp]
            
            # Apply rainfall filter  
            if max_rainfall is not None:
                filtered_weather_data = filtered_weather_data[filtered_weather_data['rainfall_mm'] <= max_rainfall]
            
            # Apply advanced weather filters
            if min_sunshine_score is not None:
                filtered_weather_data = filtered_weather_data[filtered_weather_data['sunshine_score'] >= min_sunshine_score]
            
            if max_uv_index is not None:
                filtered_weather_data = filtered_weather_data[filtered_weather_data['uv_index_max'] <= max_uv_index]
            
            if max_precip_probability is not None:
                filtered_weather_data = filtered_weather_data[filtered_weather_data['precipitation_probability_max'] <= max_precip_probability]
            
            # Apply categorical filters
            if sunshine_category and sunshine_category != "any":
                filtered_weather_data = filtered_weather_data[filtered_weather_data['sunshine_category'] == sunshine_category]
            
            if rain_category and rain_category != "any":
                filtered_weather_data = filtered_weather_data[filtered_weather_data['rain_category'] == rain_category]
            
            # Get valid cities and apply filter with fallback logic
            if len(filtered_weather_data) == 0 and len(weather_data) > 0:
                # No exact matches - find closest alternatives
                original_data = weather_data.copy()
                if travel_month is not None:
                    original_data = original_data[original_data['travel_month'] == travel_month]
                
                closest_matches = []
                
                # Find closest temperature match
                if min_temp is not None:
                    closest_temp = original_data[original_data['temp_min_c'] < min_temp]['temp_min_c'].max()
                    if pd.notna(closest_temp):
                        temp_match = original_data[original_data['temp_min_c'] == closest_temp]
                        closest_matches.append(f"Closest temperature match: {closest_temp}°C minimum")
                        filtered_weather_data = temp_match.head(3)  # Show top 3 closest
                
                # Store closest match info for user feedback
                if closest_matches:
                    filter_params['closest_match'] = "; ".join(closest_matches)
            
            valid_cities = filtered_weather_data['city'].unique()
            self.current_cities = self.current_cities[self.current_cities['city'].isin(valid_cities)]
            
        except FileNotFoundError:
            pass  # Keep all cities if no weather data available
    
    def _apply_budget_filter(self, filter_params):
        """Apply budget filter logic using real Numbeo cost data"""
        max_budget = filter_params.get("max_budget")
        budget_type = filter_params.get("budget_type", "total")
        purpose = filter_params.get("purpose", "short_stay")
        duration_days = filter_params.get("duration_days", 7)
        
        if max_budget == "no_limit":
            return  # No filtering needed
        
        # Use real cost data from Numbeo to estimate trip costs
        # For budget filtering, we'll create a cost estimate based on real prices
        
        if budget_type == "total":
            # Estimate total daily cost using real data from CSV
            # Formula: meals + transport + accommodation estimate
            self.current_cities = self.current_cities.copy()
            
            # Calculate estimated daily cost for each city
            daily_costs = []
            for _, city in self.current_cities.iterrows():
                # Daily food cost (3 meals: 2 inexpensive + 1 mid-range for 2 people / 2)
                food_cost = (2 * city.get('meal_inexpensive', 15) + 
                           city.get('meal_mid_range_2p', 60) / 2)
                
                # Daily transport cost (2 tickets)
                transport_cost = 2 * city.get('transport_ticket', 3)
                
                # Daily accommodation estimate using real apartment data
                if purpose == "short_stay":
                    # Short-term: estimate based on apartment costs (hotels ~2x apartment costs)
                    apt_1br_center = city.get('apartment_1br_center', None)
                    apt_1br_outside = city.get('apartment_1br_outside', None)
                    
                    if apt_1br_center and apt_1br_outside:
                        # Use average of center/outside, convert monthly to daily, apply hotel premium
                        monthly_rent = (apt_1br_center + apt_1br_outside) / 2
                        daily_accommodation = (monthly_rent / 30) * 2.5  # Hotel premium
                    else:
                        # Fallback to cost_index
                        cost_idx = city.get('cost_index', 70)
                        daily_accommodation = cost_idx * 0.8
                else:
                    # Long-term: use actual apartment costs
                    apt_1br_outside = city.get('apartment_1br_outside', None)
                    if apt_1br_outside:
                        daily_accommodation = apt_1br_outside / 30  # Monthly to daily
                    else:
                        # Fallback to cost_index  
                        cost_idx = city.get('cost_index', 70)
                        daily_accommodation = cost_idx * 0.5
                
                total_daily = food_cost + transport_cost + daily_accommodation
                daily_costs.append(total_daily)
            
            self.current_cities['estimated_daily_cost'] = daily_costs
            
            # Filter by total budget
            max_daily_budget = max_budget / duration_days
            self.current_cities = self.current_cities[
                self.current_cities['estimated_daily_cost'] <= max_daily_budget
            ]
            
        elif budget_type == "accommodation":
            # Use real apartment data for accommodation filtering
            self.current_cities = self.current_cities.copy()
            
            if purpose == "short_stay":
                # For short stays, estimate hotel costs from apartment data
                acceptable_cities = []
                for _, city in self.current_cities.iterrows():
                    apt_1br_center = city.get('apartment_1br_center', None)
                    apt_1br_outside = city.get('apartment_1br_outside', None)
                    
                    if apt_1br_center and apt_1br_outside:
                        # Estimate hotel cost: average apartment cost / 30 * 2.5 (hotel premium)
                        monthly_rent = (apt_1br_center + apt_1br_outside) / 2
                        estimated_hotel_cost = (monthly_rent / 30) * 2.5
                    else:
                        # Fallback: use cost_index estimation
                        cost_idx = city.get('cost_index', 70)
                        estimated_hotel_cost = cost_idx * 0.8
                    
                    if estimated_hotel_cost <= max_budget:
                        acceptable_cities.append(city.name)
                
                self.current_cities = self.current_cities.loc[acceptable_cities]
            else:
                # For long stays, use apartment rent directly
                # Filter cities where monthly apartment rent is within budget
                monthly_budget = max_budget * 30  # Convert daily to monthly
                
                apt_filter = (
                    (self.current_cities['apartment_1br_outside'].fillna(float('inf')) <= monthly_budget) |
                    (self.current_cities['apartment_1br_center'].fillna(float('inf')) <= monthly_budget)
                )
                self.current_cities = self.current_cities[apt_filter]
            
        else:  # transport - use flight cost estimates based on city cost level
            # More expensive cities tend to have more expensive flights
            # Use cost_index as proxy for transport costs
            if max_budget <= 200:
                max_cost_index = 60  # Budget destinations
            elif max_budget <= 500:
                max_cost_index = 80  # Mid-range destinations  
            else:
                max_cost_index = 120  # Any destination
                
            self.current_cities = self.current_cities[
                self.current_cities['cost_index'] <= max_cost_index
            ]
    
    def _apply_safety_filter(self, filter_params):
        """Apply safety filter logic"""
        min_safety_score = filter_params.get("min_safety_score")
        if min_safety_score != "no_requirement":
            before_filter = self.current_cities.copy()
            filtered = self.current_cities[self.current_cities['safety_score'] >= min_safety_score]
            
            if len(filtered) == 0 and len(before_filter) > 0:
                # No cities match - find closest (highest value below the requirement)
                valid_data = before_filter[before_filter['safety_score'].notna()]
                if len(valid_data) > 0:
                    closest_value = valid_data['safety_score'].max()
                    closest_city = valid_data.loc[valid_data['safety_score'] == closest_value, 'city'].iloc[0]
                    self.current_cities = valid_data[valid_data['safety_score'] == closest_value].head(1)
                    filter_params['closest_match'] = f"Closest match: {closest_city} (safety score: {closest_value})"
                else:
                    self.current_cities = filtered
            else:
                self.current_cities = filtered
    
    def _apply_language_filter(self, filter_params):
        """Apply language filter logic"""
        max_language_barrier = filter_params.get("max_language_barrier")
        if max_language_barrier != "no_preference":
            self.current_cities = self.current_cities[self.current_cities['language_barrier'] <= max_language_barrier]
    
    def _apply_healthcare_filter(self, filter_params):
        """Apply healthcare filter logic"""
        min_healthcare_score = filter_params.get("min_healthcare_score")
        if min_healthcare_score != "no_requirement":
            self.current_cities = self.current_cities[self.current_cities['healthcare_score'] >= min_healthcare_score]
    
    def _apply_pollution_filter(self, filter_params):
        """Apply pollution filter logic"""
        max_pollution_index = filter_params.get("max_pollution_index")
        if max_pollution_index != "no_concern":
            self.current_cities = self.current_cities[self.current_cities['pollution_index'] <= max_pollution_index]
    
    def _apply_public_transport_filter(self, filter_params):
        """Apply public transport filter logic"""
        min_transport_score = filter_params.get("min_transport_score")
        if min_transport_score != "no_requirement":
            self.current_cities = self.current_cities[self.current_cities['public_transport_score'] >= min_transport_score]
    
    def _apply_urban_nature_filter(self, filter_params):
        """Apply urban nature filter logic"""
        nature_preference = filter_params.get("nature_preference")
        if nature_preference != "no_preference":
            self.current_cities = self.current_cities[self.current_cities['nature_access'] == nature_preference]
    
    def _apply_city_size_filter(self, filter_params):
        """Apply city size filter logic"""
        preferred_size = filter_params.get("preferred_size")
        if preferred_size != "no_preference":
            self.current_cities = self.current_cities[self.current_cities['city_size'] == preferred_size]
    
    def _apply_walkability_filter(self, filter_params):
        """Apply walkability filter logic"""
        min_walkability_score = filter_params.get("min_walkability_score")
        if min_walkability_score != "no_requirement":
            self.current_cities = self.current_cities[self.current_cities['walkability_score'] >= min_walkability_score]
    
    def _apply_item_costs_filter(self, filter_params):
        """Apply item costs filter logic using real Numbeo pricing data"""
        filters_applied = []
        closest_matches = []
        
        # Store original cities count before filtering
        original_count = len(self.current_cities)
        
        # Helper function to apply filter with fallback to closest match
        def apply_filter_with_fallback(column_name, max_value, filter_desc):
            if max_value != -999 and max_value is not None:
                before_filter = self.current_cities.copy()
                filtered = self.current_cities[
                    self.current_cities[column_name].fillna(float('inf')) <= max_value
                ]
                
                if len(filtered) == 0 and len(before_filter) > 0:
                    # No cities match - find closest (lowest value above the limit)
                    valid_data = before_filter[before_filter[column_name].notna()]
                    if len(valid_data) > 0:
                        closest_value = valid_data[column_name].min()
                        closest_city = valid_data.loc[valid_data[column_name] == closest_value, 'city'].iloc[0]
                        closest_matches.append(f"Closest match: {closest_city} ({column_name.replace('_', ' ')}: €{closest_value:.1f})")
                        # Keep the closest city as the result
                        self.current_cities = valid_data[valid_data[column_name] == closest_value].head(1)
                    filters_applied.append(f"{filter_desc} (showing closest match)")
                else:
                    self.current_cities = filtered
                    filters_applied.append(filter_desc)
        
        # Apply each filter with fallback logic
        apply_filter_with_fallback('meal_inexpensive', filter_params.get("meal_inexpensive_max_price"), 
                                 f"inexpensive meals ≤€{filter_params.get('meal_inexpensive_max_price')}")
        
        apply_filter_with_fallback('meal_mid_range_2p', filter_params.get("meal_midrange_max_price"),
                                 f"mid-range meals ≤€{filter_params.get('meal_midrange_max_price')} (2 people)")
        
        apply_filter_with_fallback('cappuccino', filter_params.get("cappuccino_max_price"),
                                 f"cappuccino ≤€{filter_params.get('cappuccino_max_price')}")
        
        apply_filter_with_fallback('domestic_beer', filter_params.get("beer_max_price"),
                                 f"beer ≤€{filter_params.get('beer_max_price')}")
        
        apply_filter_with_fallback('taxi_1mile', filter_params.get("taxi_1mile_max_price"),
                                 f"taxi ≤€{filter_params.get('taxi_1mile_max_price')}/mile")
        
        apply_filter_with_fallback('apartment_1br_outside', filter_params.get("apartment_1br_max_price"),
                                 f"1BR apartment ≤€{filter_params.get('apartment_1br_max_price')}/month")
        
        apply_filter_with_fallback('apartment_1br_center', filter_params.get("apartment_center_max_price"),
                                 f"central 1BR ≤€{filter_params.get('apartment_center_max_price')}/month")
        
        # Add accommodation cost filters
        apply_filter_with_fallback('accommodation_entire_place_eur', filter_params.get("accommodation_entire_place_max_price"),
                                 f"entire place accommodation ≤€{filter_params.get('accommodation_entire_place_max_price')}/night")
        
        apply_filter_with_fallback('accommodation_private_room_eur', filter_params.get("accommodation_private_room_max_price"),
                                 f"private room accommodation ≤€{filter_params.get('accommodation_private_room_max_price')}/night")
        
        apply_filter_with_fallback('accommodation_avg_eur', filter_params.get("accommodation_avg_max_price"),
                                 f"average accommodation ≤€{filter_params.get('accommodation_avg_max_price')}/night")
        
        # Store applied filters and closest matches for description
        filter_params['filters_applied'] = filters_applied
        filter_params['closest_matches'] = closest_matches
        
    def get_state_summary(self) -> dict:
        """Get current search state summary with city coordinates for mapping"""
        
        # Load saved flight data
        flight_data = load_flight_data()
        
        # City coordinates for mapping (approximate city centers)
        city_coordinates = {
            "Berlin": {"lat": 52.5200, "lng": 13.4050},
            "Amsterdam": {"lat": 52.3676, "lng": 4.9041},
            "Barcelona": {"lat": 41.3851, "lng": 2.1734},
            "Prague": {"lat": 50.0755, "lng": 14.4378},
            "Lisbon": {"lat": 38.7223, "lng": -9.1393},
            "Vienna": {"lat": 48.2082, "lng": 16.3738},
            "Rome": {"lat": 41.9028, "lng": 12.4964},
            "Paris": {"lat": 48.8566, "lng": 2.3522},
            "Copenhagen": {"lat": 55.6761, "lng": 12.5683},
            "Stockholm": {"lat": 59.3293, "lng": 18.0686},
            "Brussels": {"lat": 50.8503, "lng": 4.3517},
            "Madrid": {"lat": 40.4168, "lng": -3.7038},
            "Munich": {"lat": 48.1351, "lng": 11.5820},
            "Zurich": {"lat": 47.3769, "lng": 8.5417},
            "Dublin": {"lat": 53.3498, "lng": -6.2603},
            "Budapest": {"lat": 47.4979, "lng": 19.0402},
            "Warsaw": {"lat": 52.2297, "lng": 21.0122},
            "Athens": {"lat": 37.9755, "lng": 23.7348},
            "Helsinki": {"lat": 60.1699, "lng": 24.9384},
            "Oslo": {"lat": 59.9139, "lng": 10.7522}
        }
        
        # Build list of remaining cities with their details
        remaining_cities_detailed = []
        cities_with_coords = []
        cities_full_data = []
        
        for _, city_row in self.current_cities.iterrows():
            city_name = city_row['city']
            
            # Get flight data for this city
            city_flight_data = get_flight_data_for_city(city_name, flight_data)
            
            # Basic city info with coordinates
            city_info = {
                "name": city_name,
                "country": getattr(city_row, 'country', None),
                "coordinates": city_coordinates.get(city_name, {"lat": 0, "lng": 0}),
                "cost_index": getattr(city_row, 'cost_index', None),
                "safety_score": getattr(city_row, 'safety_score', None)
            }
            remaining_cities_detailed.append(city_info)
            
            # Coordinates only (for map display)
            if city_name in city_coordinates:
                cities_with_coords.append({
                    "name": city_name,
                    "coordinates": city_coordinates[city_name]
                })
            
            # Full CSV data (for detailed analysis)
            city_data = city_row.to_dict()
            # Clean up NaN values
            city_data_clean = {k: v for k, v in city_data.items() if pd.notna(v) and v != ''}
            # Add coordinates
            city_data_clean["coordinates"] = city_coordinates.get(city_name, {"lat": 0, "lng": 0})
            # Add flight data if available
            if city_flight_data:
                city_data_clean["flight_data"] = city_flight_data
            cities_full_data.append(city_data_clean)
        
        return {
            "total_cities": len(self.all_cities),
            "remaining_cities": len(self.current_cities),
            "applied_filters": self.applied_filters,
            "remaining_city_names": self.current_cities['city'].tolist() if len(self.current_cities) <= 10 else [],
            "cities": remaining_cities_detailed,  # Basic city info with coordinates
            "cities_with_coordinates": cities_with_coords,  # Just coordinates for maps
            "cities_full_data": cities_full_data  # Complete CSV data with coordinates and flight data
        }

# FILTER TOOL SCHEMAS - LangChain compatible with required fields and sentinel values

class BudgetFilterArgs(LangChainBaseModel):
    """Schema for budget filtering arguments"""
    purpose: str = LangChainField(
        ...,
        description="Purpose of trip: 'short_stay' for weekend/vacation trips, 'long_stay' for relocation/extended stays, or 'unspecified' if user doesn't mention duration."
    )
    max_budget: int = LangChainField(
        ...,
        description="Maximum budget in EUR (e.g., 500 for '€500 budget', 1000 for '€1000 total'). Use -999 if user doesn't specify a budget amount."
    )
    budget_type: str = LangChainField(
        ...,
        description="Type of budget: 'total' for overall trip cost, 'transport' for flights only, 'accommodation' for daily lodging cost (not monthly). Use 'total' if unspecified."
    )
    duration_days: int = LangChainField(
        ...,
        description="Trip duration in days for calculation (e.g., 3 for weekend, 7 for week, 30 for month). Use -999 if not specified."
    )

class ClimateFilterArgs(LangChainBaseModel):
    """Schema for comprehensive climate filtering arguments"""
    min_temp: int = LangChainField(
        ...,
        description="Minimum temperature in Celsius (e.g., 15 for 'at least 15 degrees' or 'between 15-25 degrees'). Use -999 if user doesn't specify a minimum temperature."
    )
    max_temp: int = LangChainField(
        ...,
        description="Maximum temperature in Celsius (e.g., 25 for 'no more than 25 degrees' or 'between 15-25 degrees'). Use -999 if user doesn't specify a maximum temperature."
    )
    max_rainfall: int = LangChainField(
        ...,
        description="Maximum weekly rainfall in millimeters (e.g., 50 for 'not too rainy', 20 for 'dry weather', 100 for 'some rain ok'). Use -999 if user doesn't mention rainfall preferences."
    )
    travel_month: int = LangChainField(
        ...,
        description="Month of travel (1-12, e.g., 6 for June, 12 for December). Use -999 if user doesn't specify when they're traveling."
    )
    min_sunshine_score: int = LangChainField(
        ...,
        description="Minimum sunshine score 0-100 (e.g., 60 for 'sunny weather', 40 for 'some sunshine', 80 for 'very sunny'). Use -999 if user doesn't mention sunshine preferences."
    )
    max_uv_index: int = LangChainField(
        ...,
        description="Maximum UV index 0-11+ (e.g., 6 for 'moderate UV', 3 for 'low UV', 8 for 'high UV ok'). Use -999 if user doesn't mention UV concerns."
    )
    max_precip_probability: int = LangChainField(
        ...,
        description="Maximum precipitation probability percentage (e.g., 20 for 'low chance of rain', 30 for 'occasional rain ok'). Use -999 if not specified."
    )
    sunshine_category: str = LangChainField(
        ...,
        description="Sunshine preference: 'bright' (very sunny), 'mixed' (some sun), 'bleak' (overcast ok), 'any' (no preference). Use 'any' if not mentioned."
    )
    rain_category: str = LangChainField(
        ...,
        description="Rain preference: 'arid' (very dry), 'showery' (occasional rain), 'wet' (rain ok), 'any' (no preference). Use 'any' if not mentioned."
    )

class SafetyFilterArgs(LangChainBaseModel):
    """Schema for safety filtering arguments"""
    min_safety_score: int = LangChainField(
        ...,
        description="Minimum safety score 0-100 (e.g., 80 for 'very safe', 60 for 'reasonably safe', 90 for 'extremely safe'). Use -999 if user doesn't mention safety concerns."
    )

class LanguageFilterArgs(LangChainBaseModel):
    """Schema for language filtering arguments"""
    max_language_barrier: int = LangChainField(
        ...,
        description="Maximum language barrier 1-5 (1=English speaking, 2=minimal barrier, 3=moderate, 4=some barrier, 5=any language). Use -999 if user doesn't mention language preferences."
    )

class VisaFilterArgs(LangChainBaseModel):
    """Schema for visa filtering arguments"""
    passport_country: str = LangChainField(
        ...,
        description="User's passport country (e.g., 'Germany', 'USA', 'Canada'). Use 'unspecified' if user doesn't mention their nationality."
    )
    min_visa_free_days: int = LangChainField(
        ...,
        description="Minimum visa-free stay duration in days (e.g., 90 for 3 months, 180 for 6 months). Use -999 if not relevant or unspecified."
    )

class HealthcareFilterArgs(LangChainBaseModel):
    """Schema for healthcare filtering arguments"""
    min_healthcare_score: int = LangChainField(
        ...,
        description="Minimum healthcare quality score 0-100 (e.g., 80 for 'good healthcare', 90 for 'excellent healthcare'). Use -999 if user doesn't mention healthcare concerns."
    )

class PollutionFilterArgs(LangChainBaseModel):
    """Schema for pollution filtering arguments"""
    max_pollution_index: int = LangChainField(
        ...,
        description="Maximum pollution level 0-100 (20=very clean air, 40=clean air, 60=moderate, 80=some pollution). Use -999 if user doesn't mention air quality concerns."
    )

class TourismLoadFilterArgs(LangChainBaseModel):
    """Schema for tourism load filtering arguments"""
    max_tourism_ratio: float = LangChainField(
        ...,
        description="Maximum tourist density ratio (2.0=authentic/uncrowded, 3.0=moderately touristy, 4.0=touristy but manageable). Use -999.0 if user doesn't mention crowd preferences."
    )

class PublicTransportFilterArgs(LangChainBaseModel):
    """Schema for public transport filtering arguments"""
    min_transport_score: int = LangChainField(
        ...,
        description="Minimum transport quality score 0-100 (60=adequate, 70=good, 80=very good, 90=excellent). Use -999 if user doesn't mention transport preferences."
    )

class EventsFilterArgs(LangChainBaseModel):
    """Schema for events filtering arguments"""
    min_events_score: int = LangChainField(
        ...,
        description="Minimum events/cultural scene score 0-100. Use -999 if user doesn't mention cultural interests."
    )
    event_types: str = LangChainField(
        ...,
        description="Preferred event types comma-separated (e.g., 'electronic,nightlife' or 'jazz,classical' or 'museums,arts'). Use 'unspecified' if no specific interests mentioned."
    )

class UrbanNatureFilterArgs(LangChainBaseModel):
    """Schema for urban nature filtering arguments"""
    nature_preference: str = LangChainField(
        ...,
        description="Nature access preference: 'pure_urban' (concrete jungle), 'urban_parks' (city parks), 'nature_access' (easy access to nature), 'nature_immersed' (surrounded by nature). Use 'unspecified' if not mentioned."
    )

class CitySizeFilterArgs(LangChainBaseModel):
    """Schema for city size filtering arguments"""
    preferred_size: str = LangChainField(
        ...,
        description="Preferred city size: 'intimate' (<500k, walkable), 'medium' (500k-2M), 'metropolis' (>2M, full urban experience). Use 'unspecified' if not mentioned."
    )

class ArchitectureFilterArgs(LangChainBaseModel):
    """Schema for architecture filtering arguments"""
    preferred_style: str = LangChainField(
        ...,
        description="Preferred architectural style: 'historic' (medieval/baroque/classical), 'mixed' (blend of old and new), 'modern' (contemporary/glass/steel). Use 'unspecified' if not mentioned."
    )

class WalkabilityFilterArgs(LangChainBaseModel):
    """Schema for walkability filtering arguments"""
    min_walkability_score: int = LangChainField(
        ...,
        description="Minimum walkability score 0-100 (60=moderately walkable, 70=quite walkable, 80=very walkable, 90=extremely walkable). Use -999 if user doesn't mention walking preferences."
    )

class ItemCostFilterArgs(LangChainBaseModel):
    """Schema for filtering by specific cost items from Numbeo data"""
    meal_inexpensive_max_price: int = LangChainField(
        ...,
        description="Maximum price for inexpensive restaurant meal in EUR (e.g., 12 for 'under €12 meals', 20 for 'max €20 for lunch'). Use -999 if not specified."
    )
    meal_midrange_max_price: int = LangChainField(
        ...,
        description="Maximum price for mid-range restaurant meal for 2 people in EUR (e.g., 60 for 'dinner under €60 for two', 80 for 'nice meal max €80'). Use -999 if not specified."
    )
    cappuccino_max_price: int = LangChainField(
        ...,
        description="Maximum price for cappuccino in EUR (e.g., 3 for 'coffee under €3', 5 for 'cappuccino max €5'). Use -999 if not specified."
    )
    beer_max_price: int = LangChainField(
        ...,
        description="Maximum price for domestic beer (1 pint) in EUR (e.g., 4 for 'beer under €4', 6 for 'drinks max €6'). Use -999 if not specified."
    )
    taxi_1mile_max_price: int = LangChainField(
        ...,
        description="Maximum price for taxi ride per mile in EUR (e.g., 3 for 'taxi under €3/mile', 5 for 'taxi max €5/mile'). Use -999 if not specified."
    )
    apartment_1br_max_price: int = LangChainField(
        ...,
        description="Maximum monthly rent for 1-bedroom apartment outside center in EUR (e.g., 800 for 'rent under €800', 1200 for 'apartment max €1200/month'). Use -999 if not specified."
    )
    apartment_center_max_price: int = LangChainField(
        ...,
        description="Maximum monthly rent for 1-bedroom apartment in city center in EUR (e.g., 1000 for 'central apartment under €1000'). Use -999 if not specified."
    )
    accommodation_entire_place_max_price: int = LangChainField(
        ...,
        description="Maximum nightly price for entire place accommodations (Airbnb) in EUR (e.g., 150 for 'entire apartments under €150/night', 200 for 'whole place max €200'). Use -999 if not specified."
    )
    accommodation_private_room_max_price: int = LangChainField(
        ...,
        description="Maximum nightly price for private room accommodations in EUR (e.g., 80 for 'private rooms under €80/night', 100 for 'shared place max €100'). Use -999 if not specified."
    )
    accommodation_avg_max_price: int = LangChainField(
        ...,
        description="Maximum average accommodation cost per night in EUR (e.g., 120 for 'accommodation under €120/night', 150 for 'sleep costs max €150'). Use -999 if not specified."
    )

class AccommodationLookupArgs(LangChainBaseModel):
    """Schema for looking up accommodation costs for a specific city"""
    city: str = LangChainField(
        ...,
        description="City name to look up accommodation costs for (e.g., 'Berlin', 'Barcelona', 'Amsterdam'). Must be a European city."
    )
    property_type: str = LangChainField(
        "both",
        description="Type of accommodation: 'entire_place' for apartments/houses, 'private_room' for shared accommodations, or 'both' for comparison. Use 'both' unless user specifies."
    )
    checkin_days_ahead: int = LangChainField(
        7,
        description="Days ahead for check-in (1-365). Use 7 for 'next week', 14 for 'in two weeks', 30 for 'next month'. Default: 7."
    )
    stay_duration_days: int = LangChainField(
        3,
        description="Length of stay in days (1-30). Use 3 for weekend, 7 for week, 14 for two weeks, 30 for month. Default: 3."
    )
    guests: int = LangChainField(
        1,
        description="Number of guests (1-8). Use 1 for solo travel, 2 for couple, 4 for small group. Default: 1."
    )

class FlightLookupArgs(LangChainBaseModel):
    """Schema for looking up flight prices using Dohop"""
    origin: str = LangChainField(
        ...,
        description="Origin airport code (3 letters, e.g., 'BER' for Berlin, 'NYC' for New York, 'LON' for London). Must be 3 letters."
    )
    destination: str = LangChainField(
        ..., 
        description="Destination airport code (3 letters, e.g., 'KEF' for Reykjavik, 'PAR' for Paris, 'BCN' for Barcelona). Must be 3 letters."
    )
    departure_date: str = LangChainField(
        ...,
        description="Departure date in YYYY-MM-DD format (e.g., '2025-07-18' for July 18, 2025). Must be in the year 2025 or later."
    )
    return_date: str = LangChainField(
        "",
        description="Return date in YYYY-MM-DD format for round trips (e.g., '2025-07-20'). Must be in 2025 or later if specified. Leave empty for one-way flights. Default: empty (one-way)."
    )
    passengers: int = LangChainField(
        1,
        description="Number of passengers (1-9). Use 1 for solo travel, 2 for couple, 4 for family. Default: 1."
    )
    
    @validator('departure_date')
    def validate_departure_date(cls, v):
        """Ensure departure date is valid and in 2025 or later"""
        try:
            from datetime import datetime
            date_obj = datetime.strptime(v, '%Y-%m-%d')
            if date_obj.year < 2025:
                raise ValueError(f"Departure date must be in 2025 or later, got {date_obj.year}")
            return v
        except ValueError as e:
            if "does not match format" in str(e):
                raise ValueError("Departure date must be in YYYY-MM-DD format (e.g., '2025-07-18')")
            raise e
    
    @validator('return_date')
    def validate_return_date(cls, v):
        """Ensure return date is valid and in 2025 or later (if provided)"""
        if not v or v == "":  # Allow empty return date for one-way flights
            return v
        try:
            from datetime import datetime
            date_obj = datetime.strptime(v, '%Y-%m-%d')
            if date_obj.year < 2025:
                raise ValueError(f"Return date must be in 2025 or later, got {date_obj.year}")
            return v
        except ValueError as e:
            if "does not match format" in str(e):
                raise ValueError("Return date must be in YYYY-MM-DD format (e.g., '2025-07-20')")
            raise e
    
    @validator('passengers')
    def validate_passengers(cls, v):
        """Ensure passenger count is reasonable"""
        if not 1 <= v <= 9:
            raise ValueError("Number of passengers must be between 1 and 9")
        return v

class MultiCityFlightLookupArgs(LangChainBaseModel):
    """Schema for comparing flight prices to multiple destinations"""
    origin: str = LangChainField(
        ...,
        description="Origin airport code (3 letters, e.g., 'BER' for Berlin, 'NYC' for New York, 'LON' for London). Must be 3 letters."
    )
    destinations: str = LangChainField(
        ...,
        description="Comma-separated list of destination airport codes (2-4 cities, e.g., 'KEF,PAR,BCN,AMS' for Reykjavik, Paris, Barcelona, Amsterdam). Each must be 3 letters."
    )
    departure_date: str = LangChainField(
        ...,
        description="Departure date in YYYY-MM-DD format (e.g., '2025-07-18' for July 18, 2025). Must be in the year 2025 or later."
    )
    return_date: str = LangChainField(
        "",
        description="Return date in YYYY-MM-DD format for round trips (e.g., '2025-07-20'). Must be in 2025 or later if specified. Leave empty for one-way flights. Default: empty (one-way)."
    )
    flight_budget: int = LangChainField(
        ...,
        description="Maximum flight budget in EUR per person (e.g., 300 for '€300 budget', 500 for 'under €500'). Use -999 if no budget specified."
    )
    passengers: int = LangChainField(
        1,
        description="Number of passengers (1-9). Default: 1."
    )
    
    @validator('departure_date')
    def validate_departure_date(cls, v):
        """Ensure departure date is valid and in 2025 or later"""
        try:
            from datetime import datetime
            date_obj = datetime.strptime(v, '%Y-%m-%d')
            if date_obj.year < 2025:
                raise ValueError(f"Departure date must be in 2025 or later, got {date_obj.year}")
            return v
        except ValueError as e:
            if "does not match format" in str(e):
                raise ValueError("Departure date must be in YYYY-MM-DD format (e.g., '2025-07-18')")
            raise e
    
    @validator('return_date')
    def validate_return_date(cls, v):
        """Ensure return date is valid and in 2025 or later (if provided)"""
        if not v or v == "":  # Allow empty return date for one-way flights
            return v
        try:
            from datetime import datetime
            date_obj = datetime.strptime(v, '%Y-%m-%d')
            if date_obj.year < 2025:
                raise ValueError(f"Return date must be in 2025 or later, got {date_obj.year}")
            return v
        except ValueError as e:
            if "does not match format" in str(e):
                raise ValueError("Return date must be in YYYY-MM-DD format (e.g., '2025-07-20')")
            raise e
    
    @validator('passengers')
    def validate_passengers(cls, v):
        """Ensure passenger count is reasonable"""
        if not 1 <= v <= 9:
            raise ValueError("Number of passengers must be between 1 and 9")
        return v
        
    @validator('destinations')
    def validate_destinations(cls, v):
        """Ensure destinations list is valid"""
        destinations = [dest.strip().upper() for dest in v.split(',') if dest.strip()]
        if len(destinations) < 2:
            raise ValueError("Must provide at least 2 destinations")
        if len(destinations) > 4:
            raise ValueError("Maximum 4 destinations allowed")
        for dest in destinations:
            if len(dest) != 3:
                raise ValueError(f"Each destination must be 3 letters, got '{dest}'")
        return ','.join(destinations)

# FILTER TOOLS - These narrow down the city set

@tool(args_schema=BudgetFilterArgs)
def filter_by_budget(purpose: str, max_budget: int, budget_type: str, duration_days: int) -> Dict[str, Any]:
    """
    Filter cities by budget constraints and travel purpose.
    
    This tool filters cities based on the user's budget capacity and trip duration. Different budget types
    are supported to handle various travel scenarios. Only call this tool when the user mentions a specific
    budget amount or asks about costs.
    
    Parameters:
    -----------
    purpose : str
        Trip purpose affecting budget calculation:
        - "short_stay": Weekend trips, vacations (1-14 days)
        - "long_stay": Extended stays, relocations (1+ months)
        - "unspecified": When user doesn't specify duration
        
    max_budget : int
        Maximum budget in EUR. Use -999 if user doesn't specify amount.
        Examples: 500 for "€500 budget", 1000 for "€1000 total"
        
    budget_type : str
        Type of budget constraint:
        - "total": Complete trip cost including flights + accommodation
        - "transport": Flight/transport costs only
        - "accommodation": Daily lodging cost (not monthly rent)
        
    duration_days : int
        Trip duration in days for cost calculation.
        Examples: 3 for weekend, 7 for week, 30 for month
        Use -999 if not specified.
    
    Returns:
    --------
    Dict[str, Any]
        Contains filtered city count, description of applied filter, and updated search state.
    
    Examples:
    ---------
    - "€500 weekend trip" → purpose="short_stay", max_budget=500, budget_type="total", duration_days=3
    - "€2000 monthly rent" → purpose="long_stay", max_budget=2000, budget_type="accommodation", duration_days=30
    - "Under €300 for flights" → purpose="short_stay", max_budget=300, budget_type="transport", duration_days=7
    """
    
    # Convert sentinel values back to actual values for processing
    actual_purpose = "short_stay" if purpose == "unspecified" else purpose
    actual_budget_type = "total" if budget_type == "unspecified" else budget_type
    actual_duration_days = 3 if duration_days == -999 else duration_days
    
    state = search_states.get("current", SearchState())
    
    if max_budget == -999:
        # No budget constraint specified
        state.apply_filter("budget", {
            "purpose": actual_purpose, 
            "max_budget": "no_limit", 
            "budget_type": actual_budget_type,
            "duration_days": actual_duration_days
        }, None)
        
        return {
            "filtered_cities": len(state.current_cities),
            "description": f"No budget constraint specified - all {len(state.current_cities)} cities remain available",
            "state": state.get_state_summary()
        }
    
    # Apply filter through state management
    state.apply_filter("budget", {
        "purpose": actual_purpose, 
        "max_budget": max_budget, 
        "budget_type": actual_budget_type,
        "duration_days": actual_duration_days
    }, None)
    
    # Build description
    if actual_budget_type == "total":
        filter_desc = f"total budget under €{max_budget} ({actual_duration_days} days)"
    elif actual_budget_type == "accommodation":
        filter_desc = f"accommodation under €{max_budget}/night"
    else:  # transport
        filter_desc = f"transport under €{max_budget}"
    
    return {
        "filtered_cities": len(state.current_cities),
        "description": f"Filtered to {len(state.current_cities)} cities for {filter_desc}",
        "state": state.get_state_summary()
    }

@tool(args_schema=ClimateFilterArgs)
def filter_by_climate(
    min_temp: int, 
    max_temp: int, 
    max_rainfall: int, 
    travel_month: int,
    min_sunshine_score: int,
    max_uv_index: int,
    max_precip_probability: int,
    sunshine_category: str,
    rain_category: str
) -> Dict[str, Any]:
    """
    Filter cities by comprehensive weather preferences including temperature, rainfall, sunshine, and UV levels for specific travel months.
    
    This tool now supports advanced weather filtering using real climate data from Open-Meteo for all European cities across all months.
    
    Examples of user input that should trigger this tool:
    - "between 15 and 25 degrees in June" → min_temp=15, max_temp=25, travel_month=6
    - "sunny weather in July" → min_sunshine_score=70, sunshine_category="bright", travel_month=7  
    - "not too cold in December, maybe above 10" → min_temp=10, travel_month=12
    - "dry weather for hiking" → max_rainfall=20, rain_category="arid", max_precip_probability=20
    - "low UV for sensitive skin" → max_uv_index=3
    - "good weather" without specifics → min_temp=18, max_temp=28, min_sunshine_score=60
    - "warm and sunny summer trip" → min_temp=22, min_sunshine_score=75, sunshine_category="bright"
    
    Uses comprehensive climate data with temperature ranges, rainfall, sunshine hours, UV index, and precipitation probabilities.
    """
    
    # Convert sentinel values back to None for processing
    actual_min_temp = None if min_temp == -999 else min_temp
    actual_max_temp = None if max_temp == -999 else max_temp  
    actual_max_rainfall = None if max_rainfall == -999 else max_rainfall
    actual_travel_month = None if travel_month == -999 else travel_month
    actual_min_sunshine_score = None if min_sunshine_score == -999 else min_sunshine_score
    actual_max_uv_index = None if max_uv_index == -999 else max_uv_index
    actual_max_precip_probability = None if max_precip_probability == -999 else max_precip_probability
    actual_sunshine_category = None if sunshine_category == "any" else sunshine_category
    actual_rain_category = None if rain_category == "any" else rain_category
    
    state = search_states.get("current", SearchState())
    
    # Apply comprehensive climate filter through state management
    filter_params = {
        "min_temp": actual_min_temp, 
        "max_temp": actual_max_temp, 
        "max_rainfall": actual_max_rainfall,
        "travel_month": actual_travel_month,
        "min_sunshine_score": actual_min_sunshine_score,
        "max_uv_index": actual_max_uv_index,
        "max_precip_probability": actual_max_precip_probability,
        "sunshine_category": actual_sunshine_category,
        "rain_category": actual_rain_category
    }
    
    state.apply_filter("climate", filter_params, None)
    
    # Build comprehensive description of filter criteria
    filter_parts = []
    
    # Temperature conditions
    if actual_min_temp is not None:
        filter_parts.append(f"≥{actual_min_temp}°C")
    if actual_max_temp is not None:
        filter_parts.append(f"≤{actual_max_temp}°C")
    
    # Weather conditions
    if actual_max_rainfall is not None:
        filter_parts.append(f"≤{actual_max_rainfall}mm rain/week")
    if actual_min_sunshine_score is not None:
        filter_parts.append(f"sunshine score ≥{actual_min_sunshine_score}")
    if actual_max_uv_index is not None:
        filter_parts.append(f"UV index ≤{actual_max_uv_index}")
    if actual_max_precip_probability is not None:
        filter_parts.append(f"≤{actual_max_precip_probability}% rain chance")
    
    # Categorical conditions
    if actual_sunshine_category:
        filter_parts.append(f"{actual_sunshine_category} sunshine")
    if actual_rain_category:
        filter_parts.append(f"{actual_rain_category} rainfall")
    
    climate_desc = ", ".join(filter_parts) if filter_parts else "any climate"
    
    # Add month context if specified
    month_names = ["", "January", "February", "March", "April", "May", "June", 
                   "July", "August", "September", "October", "November", "December"]
    month_desc = f" in {month_names[actual_travel_month]}" if actual_travel_month else ""
    
    # Check for closest match information
    description = f"Climate filter applied for {climate_desc}{month_desc} - found {len(state.current_cities)} suitable cities"
    closest_match = filter_params.get('closest_match')
    if closest_match:
        description += f"\n\n{closest_match}"
    
    print(f'Climate filter applied: {climate_desc}{month_desc} - {len(state.current_cities)} cities remaining')
    print(f'Remaining cities: {list(state.current_cities["city"])}')
    
    return {
        "filtered_cities": len(state.current_cities),
        "description": description,
        "state": state.get_state_summary()
    }

@tool(args_schema=SafetyFilterArgs)
def filter_by_safety(min_safety_score: int) -> Dict[str, Any]:
    """
    Filter cities by minimum safety requirements.
    
    This tool filters cities based on safety scores derived from crime statistics, political stability,
    and general security conditions. Only call this tool when the user explicitly mentions safety 
    concerns, security worries, or asks about safe destinations.
    
    Parameters:
    -----------
    min_safety_score : int
        Minimum acceptable safety score (0-100 scale, higher is safer).
        Use -999 if user doesn't specify safety requirements.
        
        Score ranges:
        - 90-100: Extremely safe (e.g., "very safe", "safest places")
        - 80-89:  Very safe (e.g., "quite safe", "safe for solo travel")
        - 70-79:  Generally safe (e.g., "reasonably safe", "safe enough")
        - 60-69:  Moderately safe (e.g., "some safety concerns")
        - 50-59:  Basic safety (e.g., "need to be careful")
        - <50:    Higher risk destinations
    
    Returns:
    --------
    Dict[str, Any]
        Contains filtered city count, description of applied filter, and updated search state.
        
    Examples:
    ---------
    - "I want somewhere very safe" → min_safety_score=85
    - "Safety is my top priority" → min_safety_score=90
    - "Safe for solo female travel" → min_safety_score=80
    - "Reasonably safe is fine" → min_safety_score=70
    - "Any safety concerns?" → min_safety_score=75
    
    Notes:
    ------
    - Safety scores are based on objective data including crime rates, political stability, 
      and healthcare infrastructure
    - Consider user's travel experience and risk tolerance when setting thresholds
    - Solo travelers (especially women) typically need higher safety scores
    """
    
    state = search_states.get("current", SearchState())
    
    if min_safety_score == -999:
        # No safety constraint specified
        state.apply_filter("safety", {"min_safety_score": "no_requirement"}, None)
        
        return {
            "filtered_cities": len(state.current_cities),
            "description": f"No safety requirements specified - all {len(state.current_cities)} cities remain available",
            "state": state.get_state_summary()
        }
    
    # Apply filter through state management
    filter_params = {"min_safety_score": min_safety_score}
    state.apply_filter("safety", filter_params, None)
    
    # Check for closest match information
    description = f"Filtered to {len(state.current_cities)} cities with safety score ≥{min_safety_score}"
    closest_match = filter_params.get('closest_match')
    if closest_match:
        description += f"\n\n{closest_match}"
    
    return {
        "filtered_cities": len(state.current_cities),
        "description": description,
        "state": state.get_state_summary()
    }

@tool(args_schema=LanguageFilterArgs)
def filter_by_language(max_language_barrier: int) -> Dict[str, Any]:
    """Filter cities by language barrier (1=English speaking, 5=significant barrier). Only call this tool if the user explicitly mentions language concerns or preferences."""
    
    if max_language_barrier == -999:
        # No language preference specified
        state = search_states.get("current", SearchState())
        filtered = state.current_cities.copy()
        state.apply_filter("language", {"max_language_barrier": "no_preference"}, filtered)
        
        return {
            "filtered_cities": len(filtered),
            "description": f"No language preferences specified - all {len(filtered)} cities remain available",
            "state": state.get_state_summary()
        }
    
    state = search_states.get("current", SearchState())
    filtered = state.current_cities[state.current_cities['language_barrier'] <= max_language_barrier]
    
    barrier_desc = {1: "English-speaking", 2: "minimal barrier", 3: "moderate barrier", 4: "some barrier", 5: "any language"}
    
    state.apply_filter("language", {"max_language_barrier": max_language_barrier}, filtered)
    
    return {
        "filtered_cities": len(filtered),
        "description": f"Filtered to {len(filtered)} cities with {barrier_desc.get(max_language_barrier, 'language barrier ≤' + str(max_language_barrier))}",
        "state": state.get_state_summary()
    }

@tool
def filter_by_visa_requirements(passport_country: str, min_visa_free_days: int) -> Dict[str, Any]:
    """Filter cities by visa requirements for passport holders (only relevant for long stays). Only call this tool if the user mentions their nationality or visa concerns."""
    
    state = search_states.get("current", SearchState())
    filtered = state.current_cities[state.current_cities['visa_free_days'] >= min_visa_free_days]
    
    state.apply_filter("visa", {"passport_country": passport_country, "min_visa_free_days": min_visa_free_days}, filtered)
    
    return {
        "filtered_cities": len(filtered),
        "description": f"Filtered to {len(filtered)} cities allowing {passport_country} passport holders ≥{min_visa_free_days} days visa-free",
        "state": state.get_state_summary()
    }

@tool(args_schema=HealthcareFilterArgs)
def filter_by_healthcare(min_healthcare_score: int) -> Dict[str, Any]:
    """Filter cities by healthcare quality score (0-100). Only call this tool if the user explicitly mentions healthcare concerns or medical needs."""
    
    if min_healthcare_score == -999:
        # No healthcare requirement specified
        state = search_states.get("current", SearchState())
        filtered = state.current_cities.copy()
        state.apply_filter("healthcare", {"min_healthcare_score": "no_requirement"}, filtered)
        
        return {
            "filtered_cities": len(filtered),
            "description": f"No healthcare requirements specified - all {len(filtered)} cities remain available",
            "state": state.get_state_summary()
        }
    
    state = search_states.get("current", SearchState())
    filtered = state.current_cities[state.current_cities['healthcare_score'] >= min_healthcare_score]
    
    state.apply_filter("healthcare", {"min_healthcare_score": min_healthcare_score}, filtered)
    
    return {
        "filtered_cities": len(filtered),
        "description": f"Filtered to {len(filtered)} cities with healthcare score ≥{min_healthcare_score}",
        "state": state.get_state_summary()
    }

@tool(args_schema=PollutionFilterArgs)
def filter_by_pollution(max_pollution_index: int) -> Dict[str, Any]:
    """Filter cities by maximum pollution level (lower is better, 0-100 scale). Only call this tool if the user explicitly mentions air quality or pollution concerns."""
    
    if max_pollution_index == -999:
        # No pollution concern specified
        state = search_states.get("current", SearchState())
        filtered = state.current_cities.copy()
        state.apply_filter("pollution", {"max_pollution_index": "no_concern"}, filtered)
        
        return {
            "filtered_cities": len(filtered),
            "description": f"No air quality concerns specified - all {len(filtered)} cities remain available",
            "state": state.get_state_summary()
        }
    
    state = search_states.get("current", SearchState())
    filtered = state.current_cities[state.current_cities['pollution_index'] <= max_pollution_index]
    
    pollution_desc = {20: "very clean air", 40: "clean air", 60: "moderate pollution", 80: "some pollution", 100: "any pollution level"}
    desc = pollution_desc.get(max_pollution_index, f"pollution ≤{max_pollution_index}")
    
    state.apply_filter("pollution", {"max_pollution_index": max_pollution_index}, filtered)
    
    return {
        "filtered_cities": len(filtered),
        "description": f"Filtered to {len(filtered)} cities with {desc}",
        "state": state.get_state_summary()
    }

@tool
def filter_by_tourism_load(max_tourism_ratio: float) -> Dict[str, Any]:
    """Filter cities by tourism load (tourist-to-resident ratio, lower means less crowded). Only call this tool if the user mentions crowding, tourist density, or wanting authentic/local experiences."""
    
    state = search_states.get("current", SearchState())
    filtered = state.current_cities[state.current_cities['tourism_load_ratio'] <= max_tourism_ratio]
    
    crowd_desc = {2.0: "authentic, uncrowded", 3.0: "moderately touristy", 4.0: "touristy but manageable", 5.0: "very touristy"}
    desc = next((v for k, v in crowd_desc.items() if max_tourism_ratio <= k), f"tourism ratio ≤{max_tourism_ratio}")
    
    state.apply_filter("tourism_load", {"max_tourism_ratio": max_tourism_ratio}, filtered)
    
    return {
        "filtered_cities": len(filtered),
        "description": f"Filtered to {len(filtered)} cities that are {desc}",
        "state": state.get_state_summary()
    }

@tool(args_schema=PublicTransportFilterArgs)
def filter_by_public_transport(min_transport_score: int) -> Dict[str, Any]:
    """Filter cities by public transport quality (0-100, higher is better). Only call this tool if the user explicitly mentions transportation, public transit, or mobility needs."""
    
    if min_transport_score == -999:
        # No transport requirement specified
        state = search_states.get("current", SearchState())
        filtered = state.current_cities.copy()
        state.apply_filter("public_transport", {"min_transport_score": "no_requirement"}, filtered)
        
        return {
            "filtered_cities": len(filtered),
            "description": f"No transport requirements specified - all {len(filtered)} cities remain available",
            "state": state.get_state_summary()
        }
    
    state = search_states.get("current", SearchState())
    filtered = state.current_cities[state.current_cities['public_transport_score'] >= min_transport_score]
    
    transport_desc = {90: "excellent", 80: "very good", 70: "good", 60: "adequate", 50: "basic"}
    desc = next((v for k, v in transport_desc.items() if min_transport_score >= k), f"score ≥{min_transport_score}")
    
    state.apply_filter("public_transport", {"min_transport_score": min_transport_score}, filtered)
    
    return {
        "filtered_cities": len(filtered),
        "description": f"Filtered to {len(filtered)} cities with {desc} public transport",
        "state": state.get_state_summary()
    }

@tool
def filter_by_events(min_events_score: int, event_types: Optional[str] = None) -> Dict[str, Any]:
    """Filter cities by events/cultural scene quality and optionally by event types. Only call this tool if the user mentions cultural activities, events, nightlife, or specific interests like music/arts."""
    
    state = search_states.get("current", SearchState())
    
    # For static properties, we don't have events_score
    # This is a placeholder that would work with subjective properties
    filtered = state.current_cities.copy()
    
    # Simulate cultural alignment based on city characteristics
    if event_types:
        cultural_bonus_cities = ["Berlin", "Amsterdam", "Barcelona", "Paris", "Vienna", "Prague", "Rome"]
        filtered = filtered[filtered['city'].isin(cultural_bonus_cities)]
    
    filter_params = {"min_events_score": min_events_score}
    if event_types:
        filter_params["event_types"] = event_types
    
    state.apply_filter("events", filter_params, filtered)
    
    return {
        "filtered_cities": len(filtered),
        "description": f"Cultural scene filter applied" + (f" for {event_types}" if event_types else "") + " (requires subjective properties for scoring)",
        "state": state.get_state_summary()
    }

@tool(args_schema=UrbanNatureFilterArgs)
def filter_by_urban_nature(nature_preference: str) -> Dict[str, Any]:
    """Filter cities by urban nature and environment preference (pure_urban, urban_parks, nature_access, nature_immersed). Only call this tool if the user mentions nature, parks, green spaces, outdoors, or environmental preferences."""
    
    if nature_preference == "unspecified":
        # No nature preference specified
        state = search_states.get("current", SearchState())
        filtered = state.current_cities.copy()
        state.apply_filter("urban_nature", {"nature_preference": "no_preference"}, filtered)
        
        return {
            "filtered_cities": len(filtered),
            "description": f"No nature preferences specified - all {len(filtered)} cities remain available",
            "state": state.get_state_summary()
        }
    
    state = search_states.get("current", SearchState())
    filtered = state.current_cities[state.current_cities['nature_access'] == nature_preference]
    
    nature_desc = {
        "pure_urban": "concrete jungle atmosphere",
        "urban_parks": "good city parks and green spaces", 
        "nature_access": "easy access to mountains/beaches/lakes",
        "nature_immersed": "surrounded by/integrated with nature"
    }
    
    state.apply_filter("urban_nature", {"nature_preference": nature_preference}, filtered)
    
    return {
        "filtered_cities": len(filtered),
        "description": f"Filtered to {len(filtered)} cities with {nature_desc.get(nature_preference, nature_preference)}",
        "state": state.get_state_summary()
    }

@tool(args_schema=CitySizeFilterArgs)
def filter_by_city_size(preferred_size: str) -> Dict[str, Any]:
    """Filter cities by size preference (intimate <500k, medium 500k-2M, metropolis >2M). Only call this tool if the user mentions city size, population, or preferences about big vs small cities."""
    
    if preferred_size == "unspecified":
        # No size preference specified
        state = search_states.get("current", SearchState())
        filtered = state.current_cities.copy()
        state.apply_filter("city_size", {"preferred_size": "no_preference"}, filtered)
        
        return {
            "filtered_cities": len(filtered),
            "description": f"No city size preferences specified - all {len(filtered)} cities remain available",
            "state": state.get_state_summary()
        }
    
    state = search_states.get("current", SearchState())
    filtered = state.current_cities[state.current_cities['city_size'] == preferred_size]
    
    size_desc = {
        "intimate": "intimate, walkable cities (<500k population)",
        "medium": "medium-sized cities (500k-2M population)",
        "metropolis": "metropolitan cities (>2M population)"
    }
    
    state.apply_filter("city_size", {"preferred_size": preferred_size}, filtered)
    
    return {
        "filtered_cities": len(filtered),
        "description": f"Filtered to {len(filtered)} {size_desc.get(preferred_size, preferred_size)} cities",
        "state": state.get_state_summary()
    }

@tool
def filter_by_architecture(preferred_style: str) -> Dict[str, Any]:
    """Filter cities by architectural style preference (historic, mixed, modern). Only call this tool if the user mentions architecture, building styles, historical sites, or aesthetic preferences."""
    
    state = search_states.get("current", SearchState())
    filtered = state.current_cities[state.current_cities['architectural_style'] == preferred_style]
    
    arch_desc = {
        "historic": "historic architecture (medieval, baroque, classical)",
        "mixed": "blend of old and new architecture",
        "modern": "contemporary, modern architecture"
    }
    
    state.apply_filter("architecture", {"preferred_style": preferred_style}, filtered)
    
    return {
        "filtered_cities": len(filtered),
        "description": f"Filtered to {len(filtered)} cities with {arch_desc.get(preferred_style, preferred_style)}",
        "state": state.get_state_summary()
    }

@tool(args_schema=WalkabilityFilterArgs)
def filter_by_walkability(min_walkability_score: int) -> Dict[str, Any]:
    """Filter cities by walkability score (0-100, higher means more walkable and pedestrian-friendly). Only call this tool if the user mentions walking, pedestrian-friendly areas, or car-free preferences."""
    
    if min_walkability_score == -999:
        # No walkability requirement specified
        state = search_states.get("current", SearchState())
        filtered = state.current_cities.copy()
        state.apply_filter("walkability", {"min_walkability_score": "no_requirement"}, filtered)
        
        return {
            "filtered_cities": len(filtered),
            "description": f"No walkability requirements specified - all {len(filtered)} cities remain available",
            "state": state.get_state_summary()
        }
    
    state = search_states.get("current", SearchState())
    filtered = state.current_cities[state.current_cities['walkability_score'] >= min_walkability_score]
    
    walk_desc = {90: "extremely walkable", 80: "very walkable", 70: "quite walkable", 60: "moderately walkable", 50: "somewhat walkable"}
    desc = next((v for k, v in walk_desc.items() if min_walkability_score >= k), f"walkability ≥{min_walkability_score}")
    
    state.apply_filter("walkability", {"min_walkability_score": min_walkability_score}, filtered)
    
    return {
        "filtered_cities": len(filtered),
        "description": f"Filtered to {len(filtered)} {desc} cities",
        "state": state.get_state_summary()
    }

@tool(args_schema=ItemCostFilterArgs)
def filter_by_item_costs(
    meal_inexpensive_max_price: int,
    meal_midrange_max_price: int,
    cappuccino_max_price: int, 
    beer_max_price: int,
    taxi_1mile_max_price: int,
    apartment_1br_max_price: int,
    apartment_center_max_price: int,
    accommodation_entire_place_max_price: int,
    accommodation_private_room_max_price: int,
    accommodation_avg_max_price: int
) -> Dict[str, Any]:
    """
    Filter cities by specific cost items from real Numbeo pricing data and live Airbnb accommodation costs.
    
    Use this tool when users mention specific price limits for individual items like:
    - "cheap meals under €15", "lunch under €12", "inexpensive restaurants max €10"
    - "nice dinner under €60 for two", "mid-range meal max €80", "restaurant for 2 people under €70"
    - "coffee under €4", "cappuccino max €3", "cheap coffee shops"  
    - "beer under €5", "drinks under €6", "cheap bars"
    - "taxi max €3 per mile", "cheap taxi rides", "taxi under €4/mile"
    - "rent under €1000", "apartment max €800/month", "cheap housing"
    - "central apartment under €1200", "city center max €1500"
    - "accommodation under €100/night", "hotels max €150", "sleep costs under €80"
    - "entire place under €200/night", "whole apartment max €180"
    - "private room under €80/night", "shared accommodation max €60"
    
    This provides much more granular cost control than the general budget filter.
    All prices are in EUR. Use -999 for items not mentioned by the user.
    
    Examples:
    - "cheap meals under €12 and coffee under €3" → meal_inexpensive_max_price=12, cappuccino_max_price=3, others=-999
    - "nice dinner for two under €60" → meal_midrange_max_price=60, others=-999
    - "taxi rides max €4 per mile" → taxi_1mile_max_price=4, others=-999
    - "rent max €800 per month" → apartment_1br_max_price=800, others=-999
    - "accommodation under €120/night" → accommodation_avg_max_price=120, others=-999
    - "entire places under €150/night" → accommodation_entire_place_max_price=150, others=-999
    - "private rooms under €70/night" → accommodation_private_room_max_price=70, others=-999
    """
    
    state = search_states.get("current", SearchState())
    
    # Build filter parameters
    filter_params = {
        "meal_inexpensive_max_price": meal_inexpensive_max_price,
        "meal_midrange_max_price": meal_midrange_max_price,
        "cappuccino_max_price": cappuccino_max_price, 
        "beer_max_price": beer_max_price,
        "taxi_1mile_max_price": taxi_1mile_max_price,
        "apartment_1br_max_price": apartment_1br_max_price,
        "apartment_center_max_price": apartment_center_max_price,
        "accommodation_entire_place_max_price": accommodation_entire_place_max_price,
        "accommodation_private_room_max_price": accommodation_private_room_max_price,
        "accommodation_avg_max_price": accommodation_avg_max_price
    }
    
    # Apply the filter
    state.apply_filter("item_costs", filter_params, None)
    
    # Build description from applied filters
    filters_applied = filter_params.get('filters_applied', [])
    closest_matches = filter_params.get('closest_matches', [])
    
    if filters_applied:
        filter_desc = ", ".join(filters_applied)
        description = f"Filtered to {len(state.current_cities)} cities with {filter_desc}"
        
        # Add closest matches information if any were found
        if closest_matches:
            closest_desc = "; ".join(closest_matches)
            description += f"\n\n{closest_desc}"
    else:
        description = f"No specific cost limits specified - all {len(state.current_cities)} cities remain available"
    
    return {
        "filtered_cities": len(state.current_cities),
        "description": description,
        "state": state.get_state_summary()
    }

@tool
def get_final_recommendations() -> Dict[str, Any]:
    """Get final ranked recommendations from current filtered cities"""
    
    state = search_states.get("current", SearchState())
    
    if len(state.current_cities) == 0:
        return {"error": "No cities match your criteria. Let's relax some constraints."}
    
    # Rank by composite score using real Numbeo data
    ranked = state.current_cities.copy()
    
    # Use available columns from real data, with fallbacks for missing columns
    safety_weight = (ranked['safety_score'].fillna(70) / 100) * 40  # Higher weight for safety
    cost_weight = ((100 - ranked['cost_index'].fillna(70)) / 100) * 35  # Higher weight for cost
    
    # Additional factors if available (with fallbacks)
    transport_weight = 0
    if 'transport_monthly' in ranked.columns:
        # Lower transport cost = higher score
        max_transport = ranked['transport_monthly'].fillna(100).max()
        transport_weight = ((max_transport - ranked['transport_monthly'].fillna(100)) / max_transport) * 15
    
    meal_weight = 0  
    if 'meal_inexpensive' in ranked.columns:
        # Lower meal cost = higher score
        max_meal = ranked['meal_inexpensive'].fillna(20).max()
        meal_weight = ((max_meal - ranked['meal_inexpensive'].fillna(20)) / max_meal) * 10
    
    ranked['composite_score'] = safety_weight + cost_weight + transport_weight + meal_weight
    
    top_cities = ranked.nlargest(min(5, len(ranked)), 'composite_score')
    
    recommendations = []
    for _, city in top_cities.iterrows():
        reasons = []
        if city['safety_score'] >= 75:
            reasons.append(f"Very safe (safety score: {city['safety_score']})")
        elif city['safety_score'] >= 60:
            reasons.append(f"Moderately safe (safety score: {city['safety_score']})")
        
        if city['cost_index'] <= 50:
            reasons.append("Very affordable")
        elif city['cost_index'] <= 70:
            reasons.append("Good value")
        
        # Add specific cost details from real data
        meal_cost = city.get('meal_inexpensive', None)
        if meal_cost:
            if meal_cost <= 12:
                reasons.append(f"Cheap dining (€{meal_cost}/meal)")
            elif meal_cost <= 20:
                reasons.append(f"Affordable dining (€{meal_cost}/meal)")
        
        transport_cost = city.get('transport_ticket', None)
        if transport_cost and transport_cost <= 2:
            reasons.append(f"Cheap transport (€{transport_cost}/ticket)")
        
        # Remove references to columns we don't have
        # if city['public_transport_score'] >= 85:
        #     reasons.append("Excellent public transport")
        # if city['walkability_score'] >= 85:
        #     reasons.append("Very walkable")
        
        # Build recommendation with available real data
        rec = {
            "city": city['city'],
            "score": round(city['composite_score'], 1),
            "reasons": reasons,
            "safety_score": city['safety_score'],
            "cost_index": city['cost_index'],
            "primary_currency": city.get('primary_currency', 'EUR'),
            # Real cost data from Numbeo
            "meal_inexpensive": city.get('meal_inexpensive'),
            "domestic_beer": city.get('domestic_beer'),
            "cappuccino": city.get('cappuccino'),
            "transport_ticket": city.get('transport_ticket'),
            "transport_monthly": city.get('transport_monthly'),
            "apartment_1br_outside": city.get('apartment_1br_outside'),
            "total_cost_items": city.get('total_cost_items', 0),
            # Accommodation costs from Airbnb data
            "accommodation_entire_place_eur": city.get('accommodation_entire_place_eur'),
            "accommodation_private_room_eur": city.get('accommodation_private_room_eur'),
            "accommodation_avg_eur": city.get('accommodation_avg_eur')
        }
        
        # Add optional fields if they exist
        if 'country' in city:
            rec['country'] = city['country']
        
        recommendations.append(rec)
    
    return {
        "recommendations": recommendations,
        "state": state.get_state_summary()
    }

# DISCOVERY TOOLS - These help determine filtering criteria

@tool
def analyze_event_preferences(event_types: str) -> Dict[str, Any]:
    """Analyze what types of events/music the user likes to help with city cultural fit"""
    
    # This would eventually connect to real event APIs, but for now simulate
    event_categories = {
        "electronic": ["Berlin", "Amsterdam", "Barcelona"],
        "rock": ["Berlin", "Prague", "Budapest"], 
        "jazz": ["Paris", "Vienna", "Copenhagen"],
        "classical": ["Vienna", "Prague", "Munich"],
        "festivals": ["Barcelona", "Budapest", "Berlin"],
        "nightlife": ["Berlin", "Amsterdam", "Barcelona"],
        "museums": ["Paris", "Vienna", "Rome"],
        "arts": ["Paris", "Barcelona", "Vienna"]
    }
    
    matching_cities = []
    for category, cities in event_categories.items():
        if category.lower() in event_types.lower():
            matching_cities.extend(cities)
    
    return {
        "event_analysis": f"Based on interest in {event_types}",
        "culturally_aligned_cities": list(set(matching_cities)),
        "suggestion": "Consider cities with strong cultural scenes matching your interests"
    }

@tool(args_schema=AccommodationLookupArgs)
def lookup_accommodation_cost(
    city: str, 
    property_type: str = "both", 
    checkin_days_ahead: int = 7, 
    stay_duration_days: int = 3, 
    guests: int = 1
) -> Dict[str, Any]:
    """
    Look up real-time accommodation costs for a specific city using live Airbnb data.
    
    This tool fetches current accommodation pricing from Airbnb for the specified city and parameters.
    Use this when users ask about accommodation costs, hotel prices, or where to stay in a specific city.
    
    Parameters:
    -----------
    city : str
        European city name (e.g., 'Berlin', 'Barcelona', 'Amsterdam')
        
    property_type : str
        Type of accommodation to search for:
        - "entire_place": Whole apartments/houses for privacy and space
        - "private_room": Private rooms in shared accommodations (budget option)
        - "both": Get pricing for both types for comparison (recommended)
        
    checkin_days_ahead : int
        Days ahead for check-in date (1-365)
        Examples: 7 for next week, 14 for two weeks, 30 for next month
        
    stay_duration_days : int
        Length of stay in days (1-30)
        Examples: 3 for weekend, 7 for week, 14 for two weeks, 30 for month
        
    guests : int
        Number of guests (1-8)
        Examples: 1 for solo, 2 for couple, 4 for small group
    
    Returns:
    --------
    Dict containing current accommodation prices in EUR, including:
    - entire_place_eur: Median price for entire places (if requested)
    - private_room_eur: Median price for private rooms (if requested)  
    - Price ranges and sample sizes for transparency
    - Travel dates and search parameters used
    
    Examples:
    ---------
    - "How much does accommodation cost in Berlin?" → city="Berlin", property_type="both"
    - "What are hotel prices in Barcelona next month?" → city="Barcelona", checkin_days_ahead=30
    - "Private room costs in Amsterdam for a week?" → city="Amsterdam", property_type="private_room", stay_duration_days=7
    - "Accommodation for 2 people in Prague?" → city="Prague", guests=2
    """
    
    try:
        # Call the async accommodation lookup tool
        args = {
            "city": city,
            "property_type": property_type,
            "checkin_days_ahead": checkin_days_ahead,
            "stay_duration_days": stay_duration_days,
            "guests": guests
        }
        
        logger.info(f"Executing tool lookup_accommodation_cost with args: {args}")
        
        # Since we're in a sync LangChain tool but need to call async function
        # Use a thread pool to run the async function safely
        import asyncio
        import concurrent.futures
        
        def run_async_in_thread():
            # Create a new event loop in the thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(tool_fetch_accommodation_cost(args))
            finally:
                loop.close()
        
        # Run the async function in a separate thread with its own event loop
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_async_in_thread)
            result = future.result(timeout=180)  # 3 minute timeout
        
        logger.info(f"Tool result received: {result}")
        
        # Check if the result contains an error
        if "error" in result:
            logger.error(f"Tool returned error: {result}")
            return result
        
        # Format the result for the chatbot
        formatted_result = {
            "city": result["city"],
            "travel_dates": result.get("travel_dates", ""),
            "guests": result.get("guests", guests),
            "stay_duration": result.get("stay_duration_days", stay_duration_days)
        }
        
        # Add pricing information
        if "entire_place_eur" in result:
            formatted_result["entire_place"] = {
                "median_price_eur": result["entire_place_eur"],
                "price_range": result.get("entire_place_range", {})
            }
        
        if "private_room_eur" in result:
            formatted_result["private_room"] = {
                "median_price_eur": result["private_room_eur"], 
                "price_range": result.get("private_room_range", {})
            }
        
        # Add summary for easy interpretation
        summary_parts = []
        if "entire_place_eur" in result:
            ep_price = result["entire_place_eur"]
            ep_range = result.get("entire_place_range", {})
            summary_parts.append(f"Entire places: €{ep_price}/night (range: €{ep_range.get('min', '?')}-{ep_range.get('max', '?')})")
        
        if "private_room_eur" in result:
            pr_price = result["private_room_eur"]
            pr_range = result.get("private_room_range", {})
            summary_parts.append(f"Private rooms: €{pr_price}/night (range: €{pr_range.get('min', '?')}-{pr_range.get('max', '?')})")
        
        formatted_result["summary"] = f"Accommodation costs in {city}: {', '.join(summary_parts)}"
        formatted_result["data_source"] = "Live Airbnb data via BrightData"
        formatted_result["last_updated"] = result.get("last_updated", "")
        
        return formatted_result
        
    except Exception as e:
        logger.error(f"Error looking up accommodation cost for {city}: {e}")
        return {
            "error": f"Failed to lookup accommodation costs for {city}: {str(e)}",
            "city": city,
            "suggestion": "Try again later or check if the city name is correct (must be a European city)"
        }

@tool(args_schema=FlightLookupArgs)
def lookup_flight_prices(
    origin: str,
    destination: str, 
    departure_date: str,
    return_date: str = "",
    passengers: int = 1
) -> Dict[str, Any]:
    """
    Look up real-time flight prices using Dohop via BrightData browser automation.
    
    This tool fetches current flight pricing from Dohop for the specified route and parameters.
    Use this when users ask about flight costs, transportation prices, or travel between cities.
    
    IMPORTANT: Always include the dohop_url in your response so users can click to see more options and book flights.
    
    Parameters:
    -----------
    origin : str
        Origin airport code (3 letters, e.g., 'BER' for Berlin, 'NYC' for New York)
        
    destination : str
        Destination airport code (3 letters, e.g., 'KEF' for Reykjavik, 'PAR' for Paris)
        
    departure_date : str
        Departure date in YYYY-MM-DD format (e.g., '2025-07-18')
        
    return_date : str
        Return date for round trips in YYYY-MM-DD format (e.g., '2025-07-20')
        Leave empty for one-way flights
        
    passengers : int
        Number of passengers (1-9, default: 1)
    
    Returns:
    --------
    Dict containing current flight prices in EUR, including:
    - cheapest_flight_eur: Lowest price found
    - average_price_eur: Average of all prices found
    - total_flights: Number of flight options
    - flights: List of individual flights with prices, airlines, times
    - dohop_url: Direct link to Dohop results page (ALWAYS mention this to users)
    - timing: Performance metrics for the search
    
    Examples:
    ---------
    - "How much are flights from Berlin to Reykjavik?" → origin="BER", destination="KEF"
    - "Flight prices Paris to Barcelona July 18th?" → origin="PAR", destination="BCN", departure_date="2025-07-18"
    - "Round trip London to Amsterdam next month?" → origin="LON", destination="AMS", return_date provided
    """
    
    try:
        # Prepare arguments for the async function
        args = {
            "origin": origin,
            "destination": destination,
            "departure_date": departure_date,
            "return_date": return_date if return_date else None,
            "passengers": passengers
        }
        
        logger.info(f"Executing flight lookup with args: {args}")
        
        # Since we're in a sync LangChain tool but need to call async function
        # Use a thread pool to run the async function safely
        import asyncio
        import concurrent.futures
        
        def run_async_in_thread():
            # Create a new event loop in the thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(lookup_dohop_flights(**args))
            finally:
                loop.close()
        
        # Run the async function in a separate thread with its own event loop
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_async_in_thread)
            result = future.result(timeout=300)  # 5 minute timeout for flight searches
        
        logger.info(f"Flight lookup result: {result.get('success', False)}")
        
        # Check if the result contains an error
        if not result.get('success', False):
            logger.error(f"Flight lookup failed: {result}")
            return {
                "error": result.get('error', 'Unknown error occurred'),
                "route": f"{origin} → {destination}",
                "suggestion": "Try again later or check if the airport codes are correct"
            }
        
        # Format the result for the chatbot
        formatted_result = {
            "route": result.get("route", f"{origin} → {destination}"),
            "origin": result.get("origin", origin),
            "destination": result.get("destination", destination),
            "departure_date": result.get("departure_date", departure_date),
            "return_date": result.get("return_date"),
            "passengers": result.get("passengers", passengers),
            "trip_type": result.get("trip_type", "one_way"),
            "stay_duration_days": result.get("stay_duration_days")
        }
        
        # Add flight statistics
        stats = result.get("statistics", {})
        formatted_result["flight_statistics"] = {
            "total_flights": stats.get("total_flights", 0),
            "cheapest_flight_eur": stats.get("cheapest_flight_eur"),
            "average_price_eur": stats.get("average_price_eur"),
            "price_range": stats.get("price_range", {})
        }
        
        # Add sample flights
        flights = result.get("flights", [])
        formatted_result["sample_flights"] = []
        for flight in flights[:3]:  # Show top 3 flights
            formatted_result["sample_flights"].append({
                "price_eur": flight.get("price_eur"),
                "airline": flight.get("airline", "Unknown"),
                "departure_time": flight.get("departure_time", "Unknown"),
                "flight_number": flight.get("flight_number", "Unknown")
            })
        
        # Add timing information
        timing = result.get("timing", {})
        formatted_result["search_time_seconds"] = timing.get("total_time", 0)
        
        # Add summary for easy interpretation
        if stats.get("cheapest_flight_eur"):
            price_min = stats["price_range"]["min"]
            price_max = stats["price_range"]["max"]
            avg_price = stats["average_price_eur"]
            
            if result.get("trip_type") == "round_trip":
                formatted_result["summary"] = f"Found {stats['total_flights']} round-trip flights from {origin} to {destination}: €{price_min}-{price_max} (avg: €{avg_price}). View and book at: {result.get('dohop_url', '')}"
            else:
                formatted_result["summary"] = f"Found {stats['total_flights']} one-way flights from {origin} to {destination}: €{price_min}-{price_max} (avg: €{avg_price}). View and book at: {result.get('dohop_url', '')}"
        else:
            formatted_result["summary"] = f"No flights found for {origin} → {destination} on {departure_date}. Try different dates at: {result.get('dohop_url', '')}"
        
        formatted_result["data_source"] = "Live flight data via Dohop & BrightData"
        formatted_result["last_updated"] = result.get("scrape_timestamp", "")
        
        # Add the Dohop URL so the LLM can mention it to users
        formatted_result["dohop_url"] = result.get("dohop_url", "")
        
        # Save flight data to persistent storage
        save_flight_data(formatted_result, origin, destination)
        
        return formatted_result
        
    except Exception as e:
        logger.error(f"Error looking up flight prices for {origin} → {destination}: {e}")
        return {
            "error": f"Failed to lookup flight prices: {str(e)}",
            "route": f"{origin} → {destination}",
            "suggestion": "Try again later or check if the airport codes are correct (must be 3-letter codes)"
        }

@tool(args_schema=MultiCityFlightLookupArgs)
def compare_flight_prices(
    origin: str,
    destinations: str, 
    departure_date: str,
    return_date: str = "",
    flight_budget: int = -999,
    passengers: int = 1
) -> Dict[str, Any]:
    """
    Compare flight prices to multiple destinations from the same origin.
    
    This tool calls the single flight lookup tool for each destination and aggregates results,
    providing a comparison of prices and identifying which destinations are within budget.
    Use this when users want to compare flight costs to multiple cities or find the cheapest option.
    
    IMPORTANT: Always include dohop URLs for each destination so users can click to see more options and book flights.
    
    Parameters:
    -----------
    origin : str
        Origin airport code (3 letters, e.g., 'BER' for Berlin, 'NYC' for New York)
    destinations : str  
        Comma-separated destination airport codes (2-4 cities, e.g., 'KEF,PAR,BCN,AMS')
    departure_date : str
        Departure date in YYYY-MM-DD format (e.g., '2025-07-18')
    return_date : str, optional
        Return date for round trips in YYYY-MM-DD format (empty for one-way)
    flight_budget : int, optional
        Maximum budget in EUR per person. Use -999 if no budget specified.
    passengers : int, optional
        Number of passengers (1-9), defaults to 1
        
    Returns:
    --------
    Dict containing comparison of flight prices for all destinations, cities within budget, 
    cheapest option, and booking URLs for each destination.
    
    Examples:
    ---------
    - "Compare flights from Berlin to Reykjavik, Paris, Barcelona under €400" 
      → origin="BER", destinations="KEF,PAR,BCN", flight_budget=400
    - "Cheapest flights from London to Amsterdam, Brussels, Copenhagen in July"
      → origin="LON", destinations="AMS,BRU,CPH", departure_date="2025-07-18"
    """
    
    try:
        # Parse destinations
        destination_list = [dest.strip().upper() for dest in destinations.split(',') if dest.strip()]
        
        logger.info(f"Comparing flight prices from {origin} to {len(destination_list)} destinations: {destination_list}")
        
        # Results storage
        flight_results = {}
        successful_lookups = []
        failed_lookups = []
        
        # Import required modules
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError, as_completed
        from tools.dohop_flight_lookup import lookup_dohop_flights
        import asyncio as asyncio_module
        
        # Look up flights for all destinations concurrently
        def run_lookup_for_destination(destination):
            """Run flight lookup for a single destination"""
            try:
                logger.info(f"Looking up flights: {origin} → {destination}")
                
                loop = asyncio_module.new_event_loop()
                asyncio_module.set_event_loop(loop)
                try:
                    return destination, loop.run_until_complete(
                        lookup_dohop_flights(
                            origin=origin,
                            destination=destination,
                            departure_date=departure_date,
                            return_date=return_date if return_date else None,
                            passengers=passengers
                        )
                    )
                finally:
                    loop.close()
                    
            except Exception as e:
                logger.error(f"Error looking up {origin} → {destination}: {e}")
                return destination, {"error": f"Lookup failed: {str(e)}"}
        
        # Submit all flight lookups concurrently
        with ThreadPoolExecutor(max_workers=len(destination_list)) as executor:
            # Submit all destinations at once
            future_to_destination = {
                executor.submit(run_lookup_for_destination, dest): dest 
                for dest in destination_list
            }
            
            # Process results as they complete
            for future in as_completed(future_to_destination, timeout=600):  # 10 minute total timeout
                try:
                    destination, raw_result = future.result()
                    
                    # Process the result similar to the single flight lookup tool
                    if 'error' in raw_result:
                        result = {
                            "error": raw_result.get('error', 'Unknown error'),
                            "suggestion": "Try again later or check if the airport codes are correct"
                        }
                    else:
                        # Format the result like the single flight lookup tool does
                        stats = raw_result.get("statistics", {})
                        formatted_result = {
                            "lookup_type": "flight_price_lookup",
                            "origin": origin,
                            "destination": destination,
                            "departure_date": departure_date,
                            "return_date": return_date,
                            "passengers": passengers,
                            "trip_type": raw_result.get("trip_type", "one_way"),
                            "flight_statistics": stats,
                            "sample_flights": raw_result.get("flights", []),
                            "data_source": "Live flight data via Dohop & BrightData",
                            "last_updated": raw_result.get("scrape_timestamp", ""),
                            "dohop_url": raw_result.get("dohop_url", "")
                        }
                        
                        # Add summary
                        if stats.get("cheapest_flight_eur"):
                            price_min = stats["price_range"]["min"]
                            price_max = stats["price_range"]["max"]
                            avg_price = stats["average_price_eur"]
                            
                            if raw_result.get("trip_type") == "round_trip":
                                formatted_result["summary"] = f"Found {stats['total_flights']} round-trip flights from {origin} to {destination}: €{price_min}-{price_max} (avg: €{avg_price}). Book at: {raw_result.get('dohop_url', '')}"
                            else:
                                formatted_result["summary"] = f"Found {stats['total_flights']} one-way flights from {origin} to {destination}: €{price_min}-{price_max} (avg: €{avg_price}). Book at: {raw_result.get('dohop_url', '')}"
                        else:
                            formatted_result["summary"] = f"No flights found for {origin} → {destination} on {departure_date}. Try different dates at: {raw_result.get('dohop_url', '')}"
                        
                        result = formatted_result
                    
                    # Process the result
                    if 'error' not in result:
                        # Extract the cheapest price
                        stats = result.get('flight_statistics')
                        cheapest_price = stats.get('cheapest_flight_eur') if stats else None
                        
                        flight_results[destination] = {
                            'destination': destination,
                            'cheapest_price_eur': cheapest_price,
                            'total_flights': stats.get('total_flights', 0) if stats else 0,
                            'average_price_eur': stats.get('average_price_eur') if stats else None,
                            'price_range': stats.get('price_range', {}) if stats else {},
                            'trip_type': result.get('trip_type', 'one_way'),
                            'dohop_url': result.get('dohop_url', ''),
                            'summary': result.get('summary', ''),
                            'sample_flights': result.get('sample_flights', [])[:2]  # Top 2 flights per destination
                        }
                        successful_lookups.append(destination)
                        logger.info(f"✅ Completed {destination}: €{cheapest_price}")
                    else:
                        flight_results[destination] = {
                            'destination': destination,
                            'error': result.get('error', 'Unknown error'),
                            'cheapest_price_eur': None
                        }
                        failed_lookups.append(destination)
                        logger.info(f"❌ Failed {destination}: {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    # Handle individual future failures
                    destination = future_to_destination[future]
                    logger.error(f"Failed to process result for {destination}: {e}")
                    flight_results[destination] = {
                        'destination': destination,
                        'error': f"Processing failed: {str(e)}",
                        'cheapest_price_eur': None
                    }
                    failed_lookups.append(destination)
        
        # Remove the old sequential processing code below
        if False:  # Disable old sequential code
            try:
                logger.info(f"Looking up flights: {origin} → {destination}")
                
                # Call the flight lookup function directly
                from tools.dohop_flight_lookup import lookup_dohop_flights
                import asyncio
                
                # Run the async flight lookup in a thread
                def run_lookup():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(
                            lookup_dohop_flights(
                                origin=origin,
                                destination=destination,
                                departure_date=departure_date,
                                return_date=return_date if return_date else None,
                                passengers=passengers
                            )
                        )
                    finally:
                        loop.close()
                
                from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(run_lookup)
                    try:
                        raw_result = future.result(timeout=300)  # 5 minute timeout
                    except FutureTimeoutError:
                        raise Exception("Flight lookup timed out after 5 minutes")
                
                # Process the result similar to the single flight lookup tool
                if 'error' in raw_result:
                    result = {
                        "error": raw_result.get('error', 'Unknown error'),
                        "suggestion": "Try again later or check if the airport codes are correct"
                    }
                else:
                    # Format the result like the single flight lookup tool does
                    stats = raw_result.get("statistics", {})  # Fixed: was "flight_statistics"
                    formatted_result = {
                        "lookup_type": "flight_price_lookup",
                        "origin": origin,
                        "destination": destination,
                        "departure_date": departure_date,
                        "return_date": return_date,
                        "passengers": passengers,
                        "trip_type": raw_result.get("trip_type", "one_way"),
                        "flight_statistics": stats,
                        "sample_flights": raw_result.get("flights", []),  # Fixed: was "sample_flights"
                        "data_source": "Live flight data via Dohop & BrightData",
                        "last_updated": raw_result.get("scrape_timestamp", ""),
                        "dohop_url": raw_result.get("dohop_url", "")
                    }
                    
                    # Add summary
                    if stats.get("cheapest_flight_eur"):
                        price_min = stats["price_range"]["min"]
                        price_max = stats["price_range"]["max"]
                        avg_price = stats["average_price_eur"]
                        
                        if raw_result.get("trip_type") == "round_trip":
                            formatted_result["summary"] = f"Found {stats['total_flights']} round-trip flights from {origin} to {destination}: €{price_min}-{price_max} (avg: €{avg_price}). Book at: {raw_result.get('dohop_url', '')}"
                        else:
                            formatted_result["summary"] = f"Found {stats['total_flights']} one-way flights from {origin} to {destination}: €{price_min}-{price_max} (avg: €{avg_price}). Book at: {raw_result.get('dohop_url', '')}"
                    else:
                        formatted_result["summary"] = f"No flights found for {origin} → {destination} on {departure_date}. Try different dates at: {raw_result.get('dohop_url', '')}"
                    
                    result = formatted_result
                if 'error' not in result:
                    # Extract the cheapest price - try multiple methods
                    stats = result.get('flight_statistics')
                    cheapest_price = None
                    
                    # Method 1: From flight_statistics
                    if stats and isinstance(stats, dict):
                        cheapest_price = stats.get('cheapest_flight_eur')
                        logger.info(f"Debug for {destination}: Got price from stats: {cheapest_price}")
                    
                    # Method 2: If stats is None, try to get it from raw_result directly
                    if cheapest_price is None and 'flight_statistics' in result:
                        raw_stats = result['flight_statistics']
                        if raw_stats and isinstance(raw_stats, dict):
                            cheapest_price = raw_stats.get('cheapest_flight_eur')
                            stats = raw_stats  # Update stats for later use
                            logger.info(f"Debug for {destination}: Got price from raw stats: {cheapest_price}")
                    
                    # Method 3: If still no price, check if we can extract from the summary
                    if cheapest_price is None:
                        summary = result.get('summary', '')
                        if '€' in summary:
                            import re
                            # Look for price pattern like €56-131 or €111-294  
                            price_match = re.search(r'€(\d+)(?:-\d+)?', summary)
                            if price_match:
                                cheapest_price = int(price_match.group(1))
                                logger.info(f"Debug for {destination}: Extracted price from summary: {cheapest_price}")
                    
                    logger.info(f"Debug for {destination}: Final cheapest_price = {cheapest_price}")
                    
                    flight_results[destination] = {
                        'destination': destination,
                        'cheapest_price_eur': cheapest_price,
                        'total_flights': stats.get('total_flights', 0) if stats else 0,
                        'average_price_eur': stats.get('average_price_eur') if stats else None,
                        'price_range': stats.get('price_range', {}) if stats else {},
                        'trip_type': result.get('trip_type', 'one_way'),
                        'dohop_url': result.get('dohop_url', ''),
                        'summary': result.get('summary', ''),
                        'sample_flights': result.get('sample_flights', [])[:2]  # Top 2 flights per destination
                    }
                    successful_lookups.append(destination)
                else:
                    flight_results[destination] = {
                        'destination': destination,
                        'error': result.get('error', 'Unknown error'),
                        'cheapest_price_eur': None
                    }
                    failed_lookups.append(destination)
                    
            except Exception as e:
                logger.error(f"Error looking up {origin} → {destination}: {e}")
                flight_results[destination] = {
                    'destination': destination,
                    'error': f"Lookup failed: {str(e)}",
                    'cheapest_price_eur': None
                }
                failed_lookups.append(destination)
        
        # Analyze results
        successful_results = [r for r in flight_results.values() if r.get('cheapest_price_eur') is not None]
        
        if not successful_results:
            return {
                "error": "No successful flight lookups",
                "failed_destinations": failed_lookups,
                "suggestion": "Try again later or check if the airport codes are correct"
            }
        
        # Sort by price (cheapest first)
        successful_results.sort(key=lambda x: x['cheapest_price_eur'] or float('inf'))
        
        # Find cities within budget
        cities_within_budget = []
        cities_over_budget = []
        
        if flight_budget != -999:
            for result in successful_results:
                if result['cheapest_price_eur'] and result['cheapest_price_eur'] <= flight_budget:
                    cities_within_budget.append({
                        'destination': result['destination'],
                        'price_eur': result['cheapest_price_eur'],
                        'dohop_url': result['dohop_url']
                    })
                else:
                    cities_over_budget.append({
                        'destination': result['destination'],
                        'price_eur': result['cheapest_price_eur'],
                        'dohop_url': result['dohop_url']
                    })
        
        # Prepare summary
        cheapest_destination = successful_results[0] if successful_results else None
        most_expensive = successful_results[-1] if successful_results else None
        
        # Create price comparison table
        price_comparison = []
        for result in successful_results:
            price_comparison.append({
                'destination': result['destination'], 
                'cheapest_price_eur': result['cheapest_price_eur'],
                'total_flights': result['total_flights'],
                'dohop_url': result['dohop_url']
            })
        
        # Format response
        formatted_result = {
            'comparison_type': 'multi_destination_flight_comparison',
            'origin': origin,
            'destinations_requested': destination_list,
            'departure_date': departure_date,
            'return_date': return_date,
            'passengers': passengers,
            'trip_type': successful_results[0]['trip_type'] if successful_results else 'one_way',
            'flight_budget_eur': flight_budget if flight_budget != -999 else None,
            
            # Summary statistics
            'total_destinations_checked': len(destination_list),
            'successful_lookups': len(successful_results),
            'failed_lookups': len(failed_lookups),
            
            # Price comparison
            'price_comparison': price_comparison,
            'cheapest_option': {
                'destination': cheapest_destination['destination'] if cheapest_destination else None,
                'price_eur': cheapest_destination['cheapest_price_eur'] if cheapest_destination else None,
                'dohop_url': cheapest_destination['dohop_url'] if cheapest_destination else None
            } if cheapest_destination else None,
            
            'most_expensive_option': {
                'destination': most_expensive['destination'] if most_expensive else None,
                'price_eur': most_expensive['cheapest_price_eur'] if most_expensive else None,
                'dohop_url': most_expensive['dohop_url'] if most_expensive else None
            } if most_expensive else None,
            
            # Budget analysis
            'cities_within_budget': cities_within_budget,
            'cities_over_budget': cities_over_budget,
            'budget_analysis_available': flight_budget != -999,
            
            # Detailed results for each destination
            'detailed_results': flight_results,
            
            # Overall summary
            'summary': None  # Will be set below
        }
        
        # Create summary text
        if cheapest_destination:
            price_range = f"€{cheapest_destination['cheapest_price_eur']}"
            if most_expensive and most_expensive != cheapest_destination:
                price_range += f"-{most_expensive['cheapest_price_eur']}"
            
            summary_parts = [
                f"Compared flights from {origin} to {len(successful_results)} destinations: {price_range}"
            ]
            
            if cheapest_destination:
                summary_parts.append(f"Cheapest: {cheapest_destination['destination']} (€{cheapest_destination['cheapest_price_eur']})")
            
            if flight_budget != -999:
                budget_count = len(cities_within_budget)
                summary_parts.append(f"{budget_count} destination{'s' if budget_count != 1 else ''} within €{flight_budget} budget")
            
            formatted_result['summary'] = '. '.join(summary_parts) + '.'
        else:
            formatted_result['summary'] = f"Flight comparison failed for all {len(destination_list)} destinations"
        
        formatted_result['data_source'] = "Live flight data via Dohop & BrightData"
        
        # Save flight data for each successful destination
        for destination, result_data in flight_results.items():
            if 'error' not in result_data and result_data.get('cheapest_price_eur') is not None:
                save_flight_data(result_data, origin, destination)
        
        return formatted_result
        
    except Exception as e:
        logger.error(f"Error in multi-city flight comparison: {e}")
        return {
            "error": f"Failed to compare flight prices: {str(e)}",
            "origin": origin,
            "destinations": destinations,
            "suggestion": "Try again later or check if the airport codes are correct (must be 3-letter codes)"
        }

# Global search state storage (in production, use Redis)
search_states = {}

class ChatOrchestrator:
    def __init__(self, redis_client=None):
        self.model_variant = "gpt-4o"
        self.model = ChatOpenAI(
            temperature=0, 
            stream_usage=True, 
            streaming=True, 
            model=self.model_variant, 
            cache=False
        )
        
        self.redis_client = redis_client
        
        # Bind all tools to the model
        self.tools = [
            #filter_by_budget,
            filter_by_climate, 
            filter_by_safety,
            filter_by_item_costs,  # New granular cost filtering
            # filter_by_language,
            # filter_by_healthcare,
            # filter_by_pollution,
            # filter_by_public_transport,
            # filter_by_urban_nature,
            filter_by_city_size,
            # filter_by_walkability,
            get_final_recommendations,
            # filter_by_visa_requirements,
            # filter_by_tourism_load,
            # filter_by_architecture,

            # Dynamic tools: 
            lookup_accommodation_cost,  # Real-time Airbnb accommodation pricing
            lookup_flight_prices,  # Real-time flight pricing via Dohop
            compare_flight_prices,  # Multi-destination flight price comparison
            analyze_event_preferences,
            filter_by_events
        ]
        
        logger.info(f"Binding {len(self.tools)} tools to model:")
        for tool in self.tools:
            logger.info(f"  - {tool.name}: {tool.description}")
            if hasattr(tool, 'args_schema') and tool.args_schema:
                logger.info(f"    Schema: {tool.args_schema.schema()}")
            else:
                logger.info(f"    No explicit schema (using function signature)")
        
        self.runnable_model = self.model.bind_tools(self.tools)
        logger.info("Tools successfully bound to model")
        
        # Test climate tool schema specifically
        climate_tool = next((t for t in self.tools if t.name == 'filter_by_climate'), None)
        if climate_tool:
            logger.info(f"Climate tool details:")
            logger.info(f"  Name: {climate_tool.name}")
            logger.info(f"  Description: {climate_tool.description}")
            logger.info(f"  Args schema: {climate_tool.args_schema}")
            if hasattr(climate_tool, 'args_schema') and climate_tool.args_schema:
                schema_dict = climate_tool.args_schema.schema()
                logger.info(f"  Full schema: {json.dumps(schema_dict, indent=2)}")
        
        # In-memory fallback if no Redis
        self.sessions: Dict[str, Dict] = {}
    
    def generate_tools_description(self) -> str:
        """Generate a formatted list of available tools from their docstrings"""
        tool_descriptions = []
        
        for tool in self.tools:
            # Get tool name and function signature
            func = tool.func if hasattr(tool, 'func') else tool
            
            # Extract first sentence from docstring
            docstring = func.__doc__ or ""
            first_sentence = docstring.split('.')[0].strip() if docstring else "No description available"
            
            # Get function signature for parameters
            import inspect
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            param_str = ', '.join(params)
            
            # Format the tool description
            tool_descriptions.append(f"- {tool.name}({param_str}) - {first_sentence}")
        
        return '\n'.join(tool_descriptions)
    
    def analyze_climate_data_ranges(self) -> str:
        """Analyze climate and safety data to provide context about realistic parameter ranges"""
        try:
            import pandas as pd
            
            # Load climate data
            climate_data = pd.read_csv("../data/cities_time_dependent_sample.csv")
            
            # Analyze temperature ranges by representative months
            monthly_temps = climate_data.groupby('travel_month')['avg_temp_c'].agg(['min', 'max', 'mean']).round(1)
            monthly_rain = climate_data.groupby('travel_month')['rainfall_mm'].agg(['min', 'max', 'mean']).round(0)
            
            # Representative months for each season
            months_map = {12: "December", 3: "March", 6: "June", 9: "September"}
            
            # Build seasonal temperature descriptions
            seasonal_temps = []
            for month_num, month_name in months_map.items():
                if month_num in monthly_temps.index:
                    temp_data = monthly_temps.loc[month_num]
                    rain_data = monthly_rain.loc[month_num]
                    seasonal_temps.append(
                        f"  - {month_name}: {temp_data['min']}°C to {temp_data['max']}°C "
                        f"(avg: {temp_data['mean']}°C), rain: {rain_data['min']}-{rain_data['max']}mm"
                    )
            
            # Overall ranges for reference
            temp_min = climate_data['avg_temp_c'].min()
            temp_max = climate_data['avg_temp_c'].max()
            rain_min = climate_data['rainfall_mm'].min()
            rain_max = climate_data['rainfall_mm'].max()
            
            # Load safety and cost data from real Numbeo data
            static_data = pd.read_csv("../data/cities_static_properties_with_accommodation.csv")
            safety_min = static_data['safety_score'].min()
            safety_max = static_data['safety_score'].max() 
            safety_avg = static_data['safety_score'].mean()
            
            cost_min = static_data['cost_index'].min()
            cost_max = static_data['cost_index'].max()
            cost_avg = static_data['cost_index'].mean()
            
            # Create safety score interpretation guide based on real Numbeo data
            safety_ranges = []
            if len(static_data) > 0:
                # Group cities by actual safety score ranges
                very_safe = static_data[static_data['safety_score'] >= 75]['city'].tolist()
                moderately_safe = static_data[(static_data['safety_score'] >= 60) & (static_data['safety_score'] < 75)]['city'].tolist()
                basic_safe = static_data[static_data['safety_score'] < 60]['city'].tolist()
                
                if very_safe:
                    safety_ranges.append(f"  - 75+: Very safe ({', '.join(very_safe[:5])}{'...' if len(very_safe) > 5 else ''})")
                if moderately_safe:
                    safety_ranges.append(f"  - 60-74: Moderately safe ({', '.join(moderately_safe[:5])}{'...' if len(moderately_safe) > 5 else ''})")
                if basic_safe:
                    safety_ranges.append(f"  - <60: Basic safety ({', '.join(basic_safe[:5])}{'...' if len(basic_safe) > 5 else ''})")
            
            # Create cost index interpretation guide based on real Numbeo data
            cost_ranges = []
            if len(static_data) > 0:
                # Group cities by actual cost index ranges
                very_affordable = static_data[static_data['cost_index'] <= 50]['city'].tolist()
                affordable = static_data[(static_data['cost_index'] > 50) & (static_data['cost_index'] <= 70)]['city'].tolist()
                moderate = static_data[(static_data['cost_index'] > 70) & (static_data['cost_index'] <= 90)]['city'].tolist()
                expensive = static_data[static_data['cost_index'] > 90]['city'].tolist()
                
                if very_affordable:
                    cost_ranges.append(f"  - ≤50: Very affordable ({', '.join(very_affordable[:3])}{'...' if len(very_affordable) > 3 else ''})")
                if affordable:
                    cost_ranges.append(f"  - 51-70: Affordable ({', '.join(affordable[:3])}{'...' if len(affordable) > 3 else ''})")
                if moderate:
                    cost_ranges.append(f"  - 71-90: Moderate cost ({', '.join(moderate[:3])}{'...' if len(moderate) > 3 else ''})")
                if expensive:
                    cost_ranges.append(f"  - 90+: Expensive ({', '.join(expensive[:3])}{'...' if len(expensive) > 3 else ''})")
            
            # Add real cost examples from Numbeo data
            if len(static_data) > 0:
                sample_costs = []
                example_cities = ['Berlin', 'Munich', 'Zurich', 'Prague', 'Budapest']
                for city in example_cities[:min(5, len(static_data))]:
                    city_data = static_data[static_data['city'] == city]
                    if not city_data.empty:
                        row = city_data.iloc[0]
                        meal_inexp = row.get('meal_inexpensive', 'N/A')
                        meal_mid = row.get('meal_mid_range_2p', 'N/A')
                        cappuccino = row.get('cappuccino', 'N/A')
                        beer = row.get('domestic_beer', 'N/A') 
                        taxi = row.get('taxi_1mile', 'N/A')
                        rent = row.get('apartment_1br_outside', 'N/A')
                        sample_costs.append(f"  - {city}: Cheap meal €{meal_inexp}, Nice dinner(2p) €{meal_mid}, Coffee €{cappuccino}, Beer €{beer}, Taxi/mile €{taxi}, Rent €{rent}")
                
                cost_ranges.extend(["", "Real cost examples (cheap meal/nice dinner/coffee/beer/taxi/rent):"] + sample_costs)
                
                # Add price ranges for the new item cost filter
                if len(static_data) > 0:
                    meal_inexp_range = f"{static_data['meal_inexpensive'].min():.0f}-{static_data['meal_inexpensive'].max():.0f}"
                    meal_mid_range = f"{static_data['meal_mid_range_2p'].min():.0f}-{static_data['meal_mid_range_2p'].max():.0f}"
                    coffee_range = f"{static_data['cappuccino'].min():.1f}-{static_data['cappuccino'].max():.1f}"
                    beer_range = f"{static_data['domestic_beer'].min():.1f}-{static_data['domestic_beer'].max():.1f}"
                    taxi_range = f"{static_data['taxi_1mile'].min():.1f}-{static_data['taxi_1mile'].max():.1f}"
                    rent_range = f"{static_data['apartment_1br_outside'].min():.0f}-{static_data['apartment_1br_outside'].max():.0f}"
                    
                    cost_ranges.extend([
                        "",
                        "Item cost ranges (for filter_by_item_costs):",
                        f"  - Inexpensive meals: €{meal_inexp_range} | Mid-range meals (2p): €{meal_mid_range}",
                        f"  - Coffee: €{coffee_range} | Beer: €{beer_range} | Taxi/mile: €{taxi_range}",
                        f"  - Rent: €{rent_range}/month"
                    ])
            
            description = f"""
DATA CONTEXT - Parameter Ranges:

CLIMATE (filter_by_climate):
• Seasonal Temperature & Rainfall Patterns:
{chr(10).join(seasonal_temps)}
• Overall ranges: Temperature {temp_min}°C to {temp_max}°C, Rainfall {rain_min}-{rain_max}mm
• Available months: 1-12 (January to December)

SAFETY (filter_by_safety):
• Safety scores: {safety_min} to {safety_max} (average: {safety_avg:.1f})
{chr(10).join(safety_ranges)}

COST FILTERING:
• Overall cost index: {cost_min} to {cost_max} (average: {cost_avg:.1f})
{chr(10).join(cost_ranges)}

TOOL SELECTION for cost filtering:
• Use filter_by_budget for: total trip budgets, daily budgets, accommodation budgets
• Use filter_by_item_costs for: specific item price limits (meals, coffee, rent, transport)

Use these ranges to transform user requirements:
- 'very hot' → 25-30°C in summer, 'mild weather' → 15-20°C in spring/fall
- 'extremely safe' → 90+, 'very safe' → 80+, 'reasonably safe' → 70+
- 'budget/cheap' → cost index ≤60, 'expensive' → cost index ≥85""".strip()
            
            logger.info(f"📊 Climate, safety, and cost data analysis completed")
            return description
            
        except Exception as e:
            logger.error(f"Failed to analyze data: {e}")
            return "Data ranges: Unable to analyze (data files not found or invalid)"
    
    async def save_session_to_redis(self, session_id: str, session_data: dict, search_state: SearchState):
        """Save session and search state to Redis"""
        if not self.redis_client:
            return
        
        try:
            # Convert messages to serializable format
            serializable_messages = []
            for msg in session_data.get("messages", []):
                if hasattr(msg, 'dict'):
                    serializable_messages.append(msg.dict())
                else:
                    # Handle different message types
                    msg_dict = {
                        "type": type(msg).__name__,
                        "content": getattr(msg, 'content', ''),
                    }
                    if hasattr(msg, 'tool_calls'):
                        msg_dict["tool_calls"] = msg.tool_calls
                    if hasattr(msg, 'tool_call_id'):
                        msg_dict["tool_call_id"] = msg.tool_call_id
                    serializable_messages.append(msg_dict)
            
            # logger.info(f"=== SAVING TO REDIS {session_id} ===")
            # logger.info(f"Session has {len(session_data.get('messages', []))} messages")
            # logger.info(f"Serialized to {len(serializable_messages)} messages")
            # for i, msg in enumerate(serializable_messages):
            #     logger.info(f"  Saving message {i}: {msg.get('type')} - {msg.get('content', '')[:50]}")
            
            # Save session data
            session_key = f"chat_session:{session_id}"
            await self.redis_client.hset(session_key, mapping={
                "messages": json.dumps(serializable_messages),
                "created_at": session_data.get("created_at", ""),
                "updated_at": pd.Timestamp.now().isoformat()
            })
            
            # Save search state
            search_key = f"search_state:{session_id}"
            search_data = {
                "applied_filters": json.dumps(search_state.applied_filters),
                "remaining_cities": json.dumps(search_state.current_cities.to_dict('records')),
                "total_cities": str(len(search_state.all_cities))
            }
            await self.redis_client.hset(search_key, mapping=search_data)
            
            # Set expiration (7 days)
            await self.redis_client.expire(session_key, 604800)
            await self.redis_client.expire(search_key, 604800)
            
            logger.info(f"Successfully saved session {session_id} to Redis")
            
        except Exception as e:
            logger.error(f"Failed to save session to Redis: {e}")
    
    async def load_session_from_redis(self, session_id: str) -> tuple[dict, SearchState]:
        """Load session and search state from Redis"""
        if not self.redis_client:
            return {}, SearchState()
        
        try:
            # Load session data
            session_key = f"chat_session:{session_id}"
            session_data = await self.redis_client.hgetall(session_key)
            
            logger.info(f"=== LOADING FROM REDIS {session_id} ===")
            logger.info(f"Raw session data from Redis: {session_data}")
            
            if not session_data:
                logger.info(f"No session found for {session_id}, creating new")
                return {}, SearchState()
            
            # Deserialize messages
            messages = []
            if session_data.get("messages"):
                message_dicts = json.loads(session_data["messages"])
                logger.info(f"Found {len(message_dicts)} messages in Redis")
                for i, msg_dict in enumerate(message_dicts):
                    logger.info(f"  Loading message {i}: {msg_dict.get('type')} - {msg_dict.get('content', '')[:50]}")
                    msg_type = msg_dict.get("type", "human")
                    if msg_type == "human":
                        messages.append(HumanMessage(content=msg_dict["content"]))
                    elif msg_type == "ai":
                        ai_msg = AIMessage(content=msg_dict["content"])
                        if msg_dict.get("tool_calls"):
                            ai_msg.tool_calls = msg_dict["tool_calls"]
                        messages.append(ai_msg)
                    elif msg_type == "tool":
                        messages.append(ToolMessage(
                            content=msg_dict["content"],
                            tool_call_id=msg_dict.get("tool_call_id", "")
                        ))
                    elif msg_type == "system":
                        messages.append(SystemMessage(content=msg_dict["content"]))
                    else:
                        logger.warning(f"Unknown message type: {msg_type}")
                
                logger.info(f"Successfully deserialized {len(messages)} messages")
            
            session = {
                "messages": messages,
                "created_at": session_data.get("created_at"),
                "updated_at": session_data.get("updated_at")
            }
            
            # Load search state
            search_key = f"search_state:{session_id}"
            search_data = await self.redis_client.hgetall(search_key)
            
            search_state = SearchState()
            if search_data:
                # Restore applied filters
                if search_data.get("applied_filters"):
                    search_state.applied_filters = json.loads(search_data["applied_filters"])
                    # Reapply all filters to rebuild current_cities
                    search_state._reapply_all_filters()
            
            logger.info(f"Loaded session {session_id} from Redis: {len(messages)} messages, {len(search_state.applied_filters)} filters")
            return session, search_state
            
        except Exception as e:
            logger.error(f"Failed to load session from Redis: {e}")
            return {}, SearchState()
    
    async def list_user_sessions(self, user_id: str = None) -> list:
        """List available chat sessions"""
        if not self.redis_client:
            return []
        
        try:
            pattern = f"chat_session:{user_id}:*" if user_id else "chat_session:*"
            session_keys = await self.redis_client.keys(pattern)
            
            sessions = []
            for key in session_keys:
                session_data = await self.redis_client.hgetall(key)
                session_id = key.replace("chat_session:", "")
                sessions.append({
                    "session_id": session_id,
                    "created_at": session_data.get("created_at"),
                    "updated_at": session_data.get("updated_at"),
                    "message_count": len(json.loads(session_data.get("messages", "[]")))
                })
            
            return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []

    async def process_message_stream(self, client_id: str, message: dict) -> AsyncGenerator[dict, None]:
        """
        Process incoming message and stream response back
        """
        
        # Check if this is an initialization message
        is_init_message = message.get("type") == "init"
        
        # Always load from Redis on init to ensure fresh session data
        if is_init_message:
            logger.info(f"=== INITIALIZING SESSION {client_id} ===")
            session, search_state = await self.load_session_from_redis(client_id)
            
            if not session.get("messages"):
                logger.info(f"No existing session found in Redis, creating new session")
                session = {
                    "messages": [],
                    "created_at": pd.Timestamp.now().isoformat()
                }
                search_state = SearchState()
            else:
                logger.info(f"=== LOADED SESSION FROM REDIS: {len(session['messages'])} messages ===")
                for i, msg in enumerate(session["messages"]):
                    msg_type = type(msg).__name__
                    content_preview = getattr(msg, 'content', '')[:100]
                    logger.info(f"  Message {i}: {msg_type} - {content_preview}")
            
            # Update in-memory session
            self.sessions[client_id] = session
            search_states[client_id] = search_state
            search_states["current"] = search_state
            
            # Send chat history to frontend if session was loaded
            if session.get("messages"):
                serialized_messages = self.serialize_messages_for_frontend(session["messages"])
                logger.info(f"=== SENDING CHAT HISTORY TO FRONTEND ===")
                logger.info(f"Serialized {len(serialized_messages)} messages for frontend:")
                # for i, msg in enumerate(serialized_messages):
                #     logger.info(f"  Frontend Message {i}: {msg['type']} - {msg['content'][:100]}")
                
                yield {
                    "type": "chat_history",
                    "messages": serialized_messages,
                    "search_state": search_state.get_state_summary()
                }
            else:
                logger.info(f"No chat history to send for session {client_id}")
            
            logger.info(f"=== INITIALIZATION COMPLETE FOR {client_id} ===")
            return
        
        # For regular messages, ensure we have the session
        if client_id not in self.sessions:
            logger.info(f"Session not found for {client_id}, creating new")
            self.sessions[client_id] = {
                "messages": [],
                "created_at": pd.Timestamp.now().isoformat()
            }
            search_states[client_id] = SearchState()
        
        search_states["current"] = search_states[client_id]
        session = self.sessions[client_id]
        
        # Add user message to session
        user_message = HumanMessage(content=message.get('content', ''))
        session["messages"].append(user_message)
        
        # Add current search state to context
        state_summary = search_states[client_id].get_state_summary()
        
        # System prompt focusing on information gain and progressive filtering
        system_prompt = ""
        # load system prompt from file
        with open("system_prompt.txt", "r") as file:
            system_prompt = file.read()
        
        # Replace placeholder with actual available tools
        tools_description = self.generate_tools_description()
        system_prompt = system_prompt.replace("{AVAILABLE_TOOLS}", tools_description)
        logger.info(f"📋 Generated tools description:\n{tools_description}")
        
        # Replace placeholder with climate data context
        climate_context = self.analyze_climate_data_ranges()
        system_prompt = system_prompt.replace("{CLIMATE_DATA_CONTEXT}", climate_context)
        logger.info(f"🌡️ Generated climate data context:\n{climate_context}")
        
        context_message = SystemMessage(content=f"""Current search state: {json.dumps(state_summary)}

{system_prompt}""")
        
        # Prepare all content for the model
        all_content = [
            context_message,
            *session["messages"]
        ]
        
        logger.info(f"Processing message from {client_id}: {message.get('content', '')[:50]}")
        
        try:
            # Stream response from the model
            response_content = ""
            complete_tool_calls = []  # Collect tool calls after streaming
            tool_args_accumulator = {}  # Accumulate arguments for valid tool calls
            
            # First pass: stream content and collect complete tool calls
            async for chunk in self.runnable_model.astream(all_content):
                if hasattr(chunk, 'content') and chunk.content:
                    #logger.info(f"Content: {chunk.content}")
                    response_content += chunk.content
                    yield {
                        "type": "stream",
                        "content": chunk.content,
                        "partial": True
                    }
                
                # Accumulate arguments from tool_call_chunks
                if hasattr(chunk, 'tool_call_chunks') and chunk.tool_call_chunks:
                    for chunk_item in chunk.tool_call_chunks:
                        if chunk_item.get('args'):  # Only accumulate if there are args
                            call_id = chunk_item.get('id') or chunk_item.get('index', 0)
                            if call_id not in tool_args_accumulator:
                                tool_args_accumulator[call_id] = ""
                            tool_args_accumulator[call_id] += chunk_item['args']
                            logger.info(f"📝 Accumulating args for call_id '{call_id}': {chunk_item['args'][:50]}...")
                            logger.info(f"🗂️  Current accumulator: {dict((k, v[:50] + '...' if len(v) > 50 else v) for k, v in tool_args_accumulator.items())}")
                
                # Just log tool calls, don't process them yet
                if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                    logger.info(f"🔎 Found tool calls chunk: {[(tc.get('name'), tc.get('id')) for tc in chunk.tool_calls]}")
                    # Accumulate all tool calls with valid names
                    for tool_call in chunk.tool_calls:
                        if tool_call.get('name'):  # Only keep tool calls with actual names
                            # Check if this tool call is already in our list (avoid duplicates)
                            existing_ids = [tc.get('id') for tc in complete_tool_calls]
                            if tool_call.get('id') not in existing_ids:
                                complete_tool_calls.append(tool_call)
                                logger.info(f"✅ Added tool call: {tool_call.get('name')} (ID: {tool_call.get('id')})")
                    
                    logger.info(f"📋 Complete tool calls so far: {[(tc.get('name'), tc.get('id')) for tc in complete_tool_calls]}")
            
            # Second pass: process complete tool calls after streaming finishes
            if complete_tool_calls:
                logger.info(f"🔧 Processing {len(complete_tool_calls)} tool calls")
                logger.info(f"🗃️  Available accumulated args: {list(tool_args_accumulator.keys())}")
                
                # Create a better mapping for multiple tool calls
                # If we have multiple accumulated args but mismatched IDs, try to match by position
                if len(complete_tool_calls) > 1 and len(tool_args_accumulator) > 1:
                    logger.info("🔄 Multiple tool calls detected - attempting position-based mapping")
                
                # Collect all tool results before generating follow-up response
                all_tool_results = []
            
            for idx, tool_call in enumerate(complete_tool_calls):
                logger.info(f"🎯 Processing tool call {idx}: {tool_call}")
                
                tool_name = tool_call.get('name', '')
                tool_args = tool_call.get('args', {})
                tool_id = tool_call.get('id')
                
                logger.info(f"🔍 Tool: {tool_name}, ID: {tool_id}, Original args: {tool_args}")
                
                # Try multiple strategies to find the right accumulated arguments
                args_str = None
                
                # Strategy 1: Direct ID match
                if tool_id in tool_args_accumulator:
                    args_str = tool_args_accumulator[tool_id]
                    logger.info(f"✅ Found args by direct ID match: {tool_id}")
                
                # Strategy 2: Position-based match (for multiple tools)
                elif idx in tool_args_accumulator:
                    args_str = tool_args_accumulator[idx]
                    logger.info(f"✅ Found args by position match: index {idx}")
                
                # Strategy 3: Index-based fallback for single tool
                elif len(complete_tool_calls) == 1 and tool_args_accumulator:
                    # For single tool call, try the first available accumulated args
                    first_key = list(tool_args_accumulator.keys())[0]
                    args_str = tool_args_accumulator[first_key]
                    logger.info(f"✅ Found args by single-tool fallback: key {first_key}")
                
                if args_str:
                    logger.info(f"🔄 Using accumulated args: '{args_str[:100]}...'")
                    try:
                        accumulated_args = json.loads(args_str)
                        logger.info(f"✅ Successfully parsed accumulated args: {accumulated_args}")
                        logger.info(f"⚠️  REPLACING original args {tool_args} with accumulated args {accumulated_args}")
                        tool_args = accumulated_args
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ Failed to parse accumulated args '{args_str}': {e}")
                else:
                    logger.info(f"ℹ️  No accumulated args found for tool_id '{tool_id}', using original: {tool_args}")
                
                if not tool_name:
                    logger.error("Tool call missing name")
                    continue
                
                logger.info(f"Executing tool {tool_name} with args: {tool_args}")
                
                # Add the tool call message to conversation context
                tool_call_message = AIMessage(
                    content="",
                    tool_calls=[{
                        "name": tool_name,
                        "args": tool_args,
                        "id": tool_call.get('id', ''),
                        "type": "tool_call"
                    }]
                )
                session["messages"].append(tool_call_message)
                
                yield {
                    "type": "tool_call",
                    "tool": tool_name,
                    "args": tool_args,
                    "status": "executing"
                }
                
                try:
                    tool_func = globals().get(tool_name)
                    if tool_func:
                        result = tool_func.invoke(tool_args)
                        logger.info(f"Tool {tool_name} result: {result}")
                        yield {
                            "type": "tool_result",
                            "tool": tool_name,
                            "result": result,
                            "status": "completed"
                        }
                        
                        # Add tool result to conversation (don't generate individual follow-up response yet)
                        tool_message = ToolMessage(
                            content=json.dumps(result),
                            tool_call_id=tool_call.get('id', '')
                        )
                        session["messages"].append(tool_message)
                        
                        # Store result for consolidated follow-up response
                        all_tool_results.append({
                            "tool_name": tool_name,
                            "result": result,
                            "tool_call": tool_call
                        })
                    else:
                        logger.error(f"Tool {tool_name} not found")
                        yield {
                            "type": "tool_result",
                            "tool": tool_name,
                            "result": {"error": f"Tool {tool_name} not found"},
                            "status": "failed"
                        }
                except Exception as e:
                    logger.error(f"Tool execution error: {e}")
                    yield {
                        "type": "tool_result",
                        "tool": tool_name,
                        "result": {"error": str(e)},
                        "status": "failed"
                    }
            
            # Generate one consolidated follow-up response for all tool results
            if complete_tool_calls and all_tool_results:
                logger.info(f"🔄 Generating consolidated follow-up response for {len(all_tool_results)} tool results")
                
                # Generate consolidated follow-up response
                followup_context_message = SystemMessage(content=f"Current search state: {json.dumps(search_states[client_id].get_state_summary())}\n\n{system_prompt}")
                followup_content = [
                    followup_context_message,
                    *session["messages"]
                ]
                
                # Add visual separator before follow-up response
                yield {
                    "type": "stream",
                    "content": "\n\n",
                    "partial": True
                }
                
                followup_response = ""
                async for followup_chunk in self.runnable_model.astream(followup_content):
                    if hasattr(followup_chunk, 'content') and followup_chunk.content:
                        logger.info(f"Consolidated follow-up content: {followup_chunk.content}")
                        followup_response += followup_chunk.content
                        yield {
                            "type": "stream", 
                            "content": followup_chunk.content,
                            "partial": True
                        }
                
                logger.info(f"Consolidated follow-up response complete. Length: {len(followup_response)}")
                
                # Add the consolidated follow-up response to session
                if followup_response.strip():
                    session["messages"].append(AIMessage(content=followup_response))
                else:
                    logger.warning("Consolidated follow-up response was empty!")
                    # Force a response if none was generated
                    tool_names = [tr["tool_name"] for tr in all_tool_results]
                    forced_response = f"I've applied {len(tool_names)} filters: {', '.join(tool_names)}. Let me know if you'd like to add more filters or if you have other preferences!"
                    session["messages"].append(AIMessage(content=forced_response))
                    yield {
                        "type": "stream",
                        "content": forced_response,
                        "partial": True
                    }
            
            logger.info(f"Finished streaming response. Final content length: {len(response_content)}")
            
            # Add assistant message to session
            if response_content.strip():
                session["messages"].append(AIMessage(content=response_content))
            
            # Save session and search state to Redis
            await self.save_session_to_redis(client_id, session, search_states[client_id])
            
            # Send final response
            final_state = search_states[client_id].get_state_summary()
            logger.info(f"Final search state: {final_state}")
            yield {
                "type": "message_complete",
                "content": response_content,
                "state": final_state
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            yield {
                "type": "error",
                "content": f"Error processing message: {str(e)}"
            }
    
    def serialize_messages_for_frontend(self, messages):
        """Convert LangChain messages to frontend-friendly format"""
        frontend_messages = []
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                frontend_messages.append({
                    "type": "user",
                    "content": msg.content,
                    "timestamp": getattr(msg, 'timestamp', None)
                })
            elif isinstance(msg, AIMessage):
                frontend_messages.append({
                    "type": "bot", 
                    "content": msg.content,
                    "tool_calls": getattr(msg, 'tool_calls', None)
                })
            elif isinstance(msg, ToolMessage):
                # Tool messages are usually internal, but we can include them for debugging
                continue
            elif isinstance(msg, SystemMessage):
                # System messages are internal
                continue
        
        return frontend_messages 