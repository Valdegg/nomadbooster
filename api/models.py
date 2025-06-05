from pydantic import BaseModel, Field, validator
from typing import List, Optional, Literal
from enum import Enum

# Enums for categorical variables
class TravelPurpose(str, Enum):
    SHORT_STAY = "short_stay"
    LONG_STAY = "long_stay"

class BudgetType(str, Enum):
    TOTAL = "total"           # Total trip cost
    TRANSPORT = "transport"   # Flight/transport only
    ACCOMMODATION = "accommodation"  # Daily accommodation cost

class LanguageBarrier(int, Enum):
    ENGLISH_NATIVE = 1        # English is widely spoken
    MINIMAL_BARRIER = 2       # Minimal language barrier
    MODERATE_BARRIER = 3      # Some English, basic communication possible
    SIGNIFICANT_BARRIER = 4   # Limited English, local language helpful
    HIGH_BARRIER = 5          # Local language required

class EventType(str, Enum):
    ELECTRONIC = "electronic"
    ROCK = "rock"
    JAZZ = "jazz"
    CLASSICAL = "classical"
    FESTIVALS = "festivals"
    NIGHTLIFE = "nightlife"
    MUSEUMS = "museums"
    ARTS = "arts"
    THEATER = "theater"
    SPORTS = "sports"

class UrbanNaturePreference(str, Enum):
    PURE_URBAN = "pure_urban"           # Concrete jungle, minimal nature needed
    URBAN_PARKS = "urban_parks"         # City with good parks and green spaces
    NATURE_ACCESS = "nature_access"     # Easy access to mountains/beaches/lakes
    NATURE_IMMERSED = "nature_immersed" # City surrounded by/integrated with nature

class CitySize(str, Enum):
    INTIMATE = "intimate"               # <500k population, walkable, intimate feel
    MEDIUM = "medium"                   # 500k-2M, good mix of amenities and manageability  
    METROPOLIS = "metropolis"           # >2M, full metropolitan experience

class ArchitecturalStyle(str, Enum):
    HISTORIC = "historic"               # Medieval, baroque, classical architecture
    MIXED = "mixed"                     # Blend of old and new
    MODERN = "modern"                   # Contemporary, glass, steel architecture

# TRAVELER CONTEXT SCHEMA

class TravelerContext(BaseModel):
    """Actionable traveler context that can be matched to city characteristics"""
    
    # Reference points for similarity matching
    favourite_place_visited: Optional[str] = Field(default=None, description="Favorite destination visited so far - used for similarity matching")
    places_to_avoid: Optional[List[str]] = Field(default=None, description="Places to exclude from recommendations")
    
    # Practical travel requirements
    group_size: Optional[int] = Field(default=1, ge=1, le=10, description="Travel group size - affects accommodation and activity recommendations")

# FILTER PARAMETER SCHEMAS

class BudgetFilter(BaseModel):
    """Budget filtering parameters with flexible budget types"""
    purpose: TravelPurpose
    budget_type: BudgetType = BudgetType.TOTAL
    max_budget: int = Field(..., gt=0, description="Maximum budget in EUR")
    duration_days: Optional[int] = Field(default=3, gt=0, description="Trip duration for calculation")

    @validator('max_budget')
    def validate_budget_range(cls, v, values):
        if 'budget_type' in values:
            if values['budget_type'] == BudgetType.TRANSPORT and v > 2000:
                raise ValueError('Transport budget seems unrealistically high')
            elif values['budget_type'] == BudgetType.ACCOMMODATION and v > 500:
                raise ValueError('Daily accommodation budget seems unrealistically high')
            elif values['budget_type'] == BudgetType.TOTAL and v > 10000:
                raise ValueError('Total budget seems unrealistically high')
        return v

class ClimateFilter(BaseModel):
    """Climate filtering parameters"""
    min_temp: Optional[int] = Field(default=None, ge=-20, le=50, description="Minimum temperature in Celsius")
    max_temp: Optional[int] = Field(default=None, ge=-20, le=50, description="Maximum temperature in Celsius")
    max_rainfall: Optional[int] = Field(default=None, ge=0, le=300, description="Maximum monthly rainfall in mm")

    @validator('max_temp')
    def validate_temp_range(cls, v, values):
        if v is not None and 'min_temp' in values and values['min_temp'] is not None:
            if v <= values['min_temp']:
                raise ValueError('max_temp must be greater than min_temp')
        return v

class SafetyFilter(BaseModel):
    """Safety filtering parameters"""
    min_safety_score: int = Field(..., ge=0, le=100, description="Minimum safety score (0-100, higher is safer)")

class LanguageFilter(BaseModel):
    """Language barrier filtering parameters"""
    max_language_barrier: LanguageBarrier = Field(..., description="Maximum acceptable language barrier")

class VisaFilter(BaseModel):
    """Visa requirements filtering parameters"""
    passport_country: str = Field(..., min_length=2, max_length=50, description="Passport issuing country")
    min_visa_free_days: int = Field(..., ge=0, le=365, description="Minimum required visa-free stay duration")

class HealthcareFilter(BaseModel):
    """Healthcare quality filtering parameters"""
    min_healthcare_score: int = Field(..., ge=0, le=100, description="Minimum healthcare score (0-100, higher is better)")

class PollutionFilter(BaseModel):
    """Pollution filtering parameters"""
    max_pollution_index: int = Field(..., ge=0, le=100, description="Maximum pollution index (0-100, lower is cleaner)")

class TourismLoadFilter(BaseModel):
    """Tourism load filtering parameters"""
    max_tourism_ratio: float = Field(..., ge=0.0, le=10.0, description="Maximum tourist-to-resident ratio (lower is more authentic)")

class PublicTransportFilter(BaseModel):
    """Public transport quality filtering parameters"""
    min_transport_score: int = Field(..., ge=0, le=100, description="Minimum transport quality score (0-100, higher is better)")

class EventsFilter(BaseModel):
    """Events and cultural scene filtering parameters"""
    min_events_score: int = Field(..., ge=0, le=100, description="Minimum events/cultural scene score (0-100)")
    event_types: Optional[List[EventType]] = Field(default=None, description="Preferred event types")

class UrbanNatureFilter(BaseModel):
    """Urban nature and environment filtering parameters"""
    nature_preference: UrbanNaturePreference = Field(..., description="Preferred level of nature access/integration")

class CitySizeFilter(BaseModel):
    """City size filtering parameters"""
    preferred_size: CitySize = Field(..., description="Preferred city size category")

class ArchitectureFilter(BaseModel):
    """Architectural style filtering parameters"""
    preferred_style: ArchitecturalStyle = Field(..., description="Preferred architectural style")

class WalkabilityFilter(BaseModel):
    """Walkability filtering parameters"""
    min_walkability_score: int = Field(..., ge=0, le=100, description="Minimum walkability score (0-100, higher means more walkable)")

# CITY DATA SCHEMAS - Organized by Data Type

class StaticCityProperties(BaseModel):
    """Properties inherent to the city that don't change with time or user preferences"""
    city: str
    country: str
    
    # Economic fundamentals
    cost_index: int = Field(..., ge=0, le=200, description="Cost of living index (100 = EU average)")
    
    # Safety and health infrastructure
    safety_score: int = Field(..., ge=0, le=100, description="Safety score (0-100, based on crime rates)")
    healthcare_score: int = Field(..., ge=0, le=100, description="Healthcare system quality (0-100)")
    
    # Communication and administration
    language_barrier: LanguageBarrier = Field(..., description="Language barrier level (1-5)")
    visa_free_days: int = Field(..., ge=0, le=365, description="Visa-free stay duration for EU citizens")
    
    # Environmental and infrastructure
    pollution_index: int = Field(..., ge=0, le=100, description="General air quality (0-100, lower is cleaner)")
    tourism_load_ratio: float = Field(..., ge=0.0, le=10.0, description="Average tourist-to-resident ratio")
    public_transport_score: int = Field(..., ge=0, le=100, description="Transport infrastructure quality (0-100)")
    
    # Urban characteristics (new filterable properties)
    nature_access: UrbanNaturePreference = Field(..., description="Level of nature access/integration")
    city_size: CitySize = Field(..., description="City size category")
    architectural_style: ArchitecturalStyle = Field(..., description="Dominant architectural style")
    walkability_score: int = Field(..., ge=0, le=100, description="Walkability score (0-100, higher is more walkable)")
    population: int = Field(..., ge=0, description="City population for size verification")

class TimeDependentCityProperties(BaseModel):
    """Properties that vary based on when you travel"""
    city: str
    
    # Climate (seasonal)
    avg_temp_c: int = Field(..., description="Average temperature for travel period in Celsius")
    rainfall_mm: int = Field(..., ge=0, description="Rainfall for travel period in millimeters")
    
    # Market prices (seasonal/demand-based)
    flight_cost_eur: int = Field(..., ge=0, description="Flight cost for travel dates in EUR")
    accommodation_cost_eur: int = Field(..., ge=0, description="Daily accommodation cost for travel dates in EUR")
    
    # Context
    travel_month: Optional[int] = Field(default=None, ge=1, le=12, description="Month of travel (1-12)")
    booking_advance_days: Optional[int] = Field(default=None, ge=0, description="Days in advance of booking")

class SubjectiveCityProperties(BaseModel):
    """Properties that depend on user preferences and interests"""
    city: str
    
    # Subjective climate assessment
    climate_rating: int = Field(..., ge=0, le=100, description="Climate desirability (subjective, 0-100)")
    
    # Cultural fit (depends on user interests)
    events_score: int = Field(..., ge=0, le=100, description="Events/cultural scene quality for user interests (0-100)")
    cultural_alignment: Optional[List[EventType]] = Field(default=None, description="Event types this city excels in")

class CompleteCityData(BaseModel):
    """Complete city data combining all three property types"""
    # Static properties (always available)
    static: StaticCityProperties
    
    # Time-dependent properties (fetched based on travel dates)
    time_dependent: Optional[TimeDependentCityProperties] = Field(default=None, description="Available when travel dates specified")
    
    # Subjective properties (calculated based on user preferences)
    subjective: Optional[SubjectiveCityProperties] = Field(default=None, description="Available when user preferences specified")
    
    @property
    def city(self) -> str:
        return self.static.city
    
    @property
    def country(self) -> str:
        return self.static.country

# RECOMMENDATION AND RESPONSE SCHEMAS

class CityRecommendation(BaseModel):
    """City recommendation with scoring explanation"""
    city: str
    country: str
    score: float = Field(..., description="Composite recommendation score")
    reasons: List[str] = Field(..., description="Reasons why this city matches preferences")
    
    # Key metrics for display
    avg_temp_c: int
    safety_score: int
    flight_cost_eur: int
    accommodation_cost_eur: int
    events_score: int
    language_barrier: LanguageBarrier
    
    # Additional context
    estimated_daily_cost: Optional[int] = Field(default=None, description="Estimated daily cost in EUR")
    visa_requirements: Optional[str] = Field(default=None, description="Visa requirements summary")

class SearchState(BaseModel):
    """Current search filtering state"""
    total_cities: int = Field(..., description="Total cities in dataset")
    remaining_cities: int = Field(..., description="Cities remaining after filters")
    applied_filters: dict = Field(default_factory=dict, description="Currently applied filters")
    remaining_city_names: List[str] = Field(default_factory=list, description="Names of remaining cities (if ≤10)")

class FilterResponse(BaseModel):
    """Response from applying a filter"""
    filtered_cities: int = Field(..., description="Number of cities remaining after filter")
    description: str = Field(..., description="Human-readable description of filter applied")
    state: SearchState = Field(..., description="Updated search state")

class RecommendationResponse(BaseModel):
    """Final recommendation response"""
    recommendations: List[CityRecommendation]
    state: SearchState
    conversation_complete: bool = Field(default=True, description="Whether search is complete")

# CHAT MESSAGE SCHEMAS

class ChatMessage(BaseModel):
    """Chat message structure"""
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: Optional[str] = None

# METRIC DEFINITIONS AND SCALES

class MetricDefinitions:
    """Documentation of all metrics organized by data type"""
    
    STATIC_METRICS = {
        "cost_index": "Cost of living index (100 = EU average, 50 = very cheap, 150 = expensive)",
        "safety_score": "Safety score (0-100, higher is safer, based on crime rates)",
        "healthcare_score": "Healthcare system quality (0-100, WHO efficiency + accessibility)",
        "language_barrier": "Communication difficulty (1=English common, 5=local language required)",
        "visa_free_days": "Days EU citizens can stay without visa (0-365)",
        "pollution_index": "General air quality (0-100, lower is cleaner air)",
        "tourism_load_ratio": "Average tourist density (1.0=authentic, 5.0=very touristy)",
        "public_transport_score": "Transport infrastructure quality (0-100, coverage + reliability)",
        "nature_access": "Nature integration level (pure_urban, urban_parks, nature_access, nature_immersed)",
        "city_size": "Population category (intimate <500k, medium 500k-2M, metropolis >2M)",
        "architectural_style": "Dominant architecture (historic, mixed, modern)",
        "walkability_score": "Walkability rating (0-100, pedestrian-friendly infrastructure)",
        "population": "City population for size verification and context"
    }
    
    TIME_DEPENDENT_METRICS = {
        "avg_temp_c": "Average temperature for travel period in Celsius (-20 to 50)",
        "rainfall_mm": "Rainfall for travel period in millimeters (0-300)",
        "flight_cost_eur": "Flight cost for specific travel dates in EUR",
        "accommodation_cost_eur": "Daily accommodation cost for travel dates in EUR",
        "travel_month": "Month of travel (1-12, affects weather and prices)",
        "booking_advance_days": "Days booking in advance (affects flight prices)"
    }
    
    SUBJECTIVE_METRICS = {
        "climate_rating": "Climate desirability based on user preferences (0-100)",
        "events_score": "Cultural scene quality for user's interests (0-100)",
        "cultural_alignment": "Event types this city excels in based on user preferences"
    }
    
    @classmethod
    def get_data_source_requirements(cls):
        """Define where each type of data comes from"""
        return {
            "static": {
                "source": "Static database/CSV",
                "update_frequency": "Rarely (annual updates)",
                "examples": ["safety ratings", "healthcare systems", "transport infrastructure"]
            },
            "time_dependent": {
                "source": "Live APIs (weather, flights, accommodation)",
                "update_frequency": "Real-time or daily",
                "examples": ["current weather", "flight prices", "hotel availability"]
            },
            "subjective": {
                "source": "Calculated based on user preferences",
                "update_frequency": "Per conversation",
                "examples": ["climate preference match", "cultural scene alignment"]
            }
        } 