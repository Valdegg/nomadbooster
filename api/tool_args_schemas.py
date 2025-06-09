from langchain_core.pydantic_v1 import BaseModel as LangChainBaseModel, Field as LangChainField, validator

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
