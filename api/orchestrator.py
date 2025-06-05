import json
import os
from typing import Dict, List, Any, AsyncGenerator, Optional
import pandas as pd
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel as LangChainBaseModel
import logging

logger = logging.getLogger(__name__)

# Global state for search space tracking
class SearchState:
    def __init__(self):
        self.all_cities = pd.read_csv("../data/cities_static_properties.csv")
        self.current_cities = self.all_cities.copy()
        self.applied_filters = {}
        
    def apply_filter(self, filter_name: str, filter_params: dict, filtered_df: pd.DataFrame):
        """Apply a filter and update state"""
        self.applied_filters[filter_name] = filter_params
        self.current_cities = filtered_df
        
    def get_state_summary(self) -> dict:
        """Get current search state summary"""
        return {
            "total_cities": len(self.all_cities),
            "remaining_cities": len(self.current_cities),
            "applied_filters": self.applied_filters,
            "remaining_city_names": self.current_cities['city'].tolist() if len(self.current_cities) <= 10 else []
        }

# FILTER TOOLS - These narrow down the city set

@tool
def filter_by_budget(purpose: str, max_budget: int, budget_type: str = "total", duration_days: int = 3) -> Dict[str, Any]:
    """Filter cities by budget constraints. Budget can be total cost, transport only, or daily accommodation cost."""
    
    state = search_states.get("current", SearchState())
    
    # For static properties CSV, we only have cost_index, not actual flight/accommodation costs
    # We'll use cost_index as a proxy: lower cost_index = cheaper city
    if budget_type == "total":
        # For total budget, prefer cities with lower cost index
        # Simple heuristic: exclude cities with cost_index > threshold based on budget
        if max_budget <= 500:
            cost_threshold = 60  # Only very affordable cities
        elif max_budget <= 1000:
            cost_threshold = 80  # Affordable to moderate cities
        elif max_budget <= 2000:
            cost_threshold = 100  # Moderate to expensive cities
        else:
            cost_threshold = 200  # All cities
            
        filtered = state.current_cities[state.current_cities['cost_index'] <= cost_threshold]
        filter_desc = f"total budget under €{max_budget} ({duration_days} days)"
        
    elif budget_type == "accommodation":
        # For accommodation budget, use cost_index as proxy for daily costs
        if max_budget <= 50:
            cost_threshold = 50
        elif max_budget <= 100:
            cost_threshold = 70
        elif max_budget <= 150:
            cost_threshold = 90
        else:
            cost_threshold = 200
            
        filtered = state.current_cities[state.current_cities['cost_index'] <= cost_threshold]
        filter_desc = f"accommodation under €{max_budget}/night"
        
    else:  # budget_type == "transport"
        # For transport-only budget, less restrictive on cost_index
        if max_budget <= 200:
            cost_threshold = 80
        elif max_budget <= 500:
            cost_threshold = 120
        else:
            cost_threshold = 200
            
        filtered = state.current_cities[state.current_cities['cost_index'] <= cost_threshold]
        filter_desc = f"transport under €{max_budget}"
    
    state.apply_filter("budget", {
        "purpose": purpose, 
        "max_budget": max_budget, 
        "budget_type": budget_type,
        "duration_days": duration_days
    }, filtered)
    
    return {
        "filtered_cities": len(filtered),
        "description": f"Filtered to {len(filtered)} cities for {filter_desc}",
        "state": state.get_state_summary()
    }

@tool  
def filter_by_climate(min_temp: Optional[int] = None, max_temp: Optional[int] = None, max_rainfall: Optional[int] = None) -> Dict[str, Any]:
    """Filter cities by temperature and rainfall preferences. Note: This uses static data, actual climate filtering requires time-dependent data."""
    
    state = search_states.get("current", SearchState())
    filtered = state.current_cities.copy()
    
    # For static properties, we don't have actual weather data
    # This is a placeholder that would work with time-dependent properties
    filter_parts = []
    if min_temp is not None:
        filter_parts.append(f"≥{min_temp}°C")
    if max_temp is not None:
        filter_parts.append(f"≤{max_temp}°C")
    if max_rainfall is not None:
        filter_parts.append(f"≤{max_rainfall}mm rain")
    
    temp_desc = ", ".join(filter_parts) if filter_parts else "any climate"
    state.apply_filter("climate", {"min_temp": min_temp, "max_temp": max_temp, "max_rainfall": max_rainfall}, filtered)
    
    return {
        "filtered_cities": len(filtered),
        "description": f"Climate filter set for {temp_desc} (requires time-dependent data for actual filtering)",
        "state": state.get_state_summary()
    }

@tool
def filter_by_safety(min_safety_score: int) -> Dict[str, Any]:
    """Filter cities by minimum safety score (0-100)"""
    
    state = search_states.get("current", SearchState())
    filtered = state.current_cities[state.current_cities['safety_score'] >= min_safety_score]
    
    state.apply_filter("safety", {"min_safety_score": min_safety_score}, filtered)
    
    return {
        "filtered_cities": len(filtered),
        "description": f"Filtered to {len(filtered)} cities with safety score ≥{min_safety_score}",
        "state": state.get_state_summary()
    }

@tool
def filter_by_language(max_language_barrier: int) -> Dict[str, Any]:
    """Filter cities by language barrier (1=English speaking, 5=significant barrier)"""
    
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
    """Filter cities by visa requirements for passport holders (only relevant for long stays)"""
    
    state = search_states.get("current", SearchState())
    filtered = state.current_cities[state.current_cities['visa_free_days'] >= min_visa_free_days]
    
    state.apply_filter("visa", {"passport_country": passport_country, "min_visa_free_days": min_visa_free_days}, filtered)
    
    return {
        "filtered_cities": len(filtered),
        "description": f"Filtered to {len(filtered)} cities allowing {passport_country} passport holders ≥{min_visa_free_days} days visa-free",
        "state": state.get_state_summary()
    }

@tool
def filter_by_healthcare(min_healthcare_score: int) -> Dict[str, Any]:
    """Filter cities by healthcare quality score (0-100)"""
    
    state = search_states.get("current", SearchState())
    filtered = state.current_cities[state.current_cities['healthcare_score'] >= min_healthcare_score]
    
    state.apply_filter("healthcare", {"min_healthcare_score": min_healthcare_score}, filtered)
    
    return {
        "filtered_cities": len(filtered),
        "description": f"Filtered to {len(filtered)} cities with healthcare score ≥{min_healthcare_score}",
        "state": state.get_state_summary()
    }

@tool
def filter_by_pollution(max_pollution_index: int) -> Dict[str, Any]:
    """Filter cities by maximum pollution level (lower is better, 0-100 scale)"""
    
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
    """Filter cities by tourism load (tourist-to-resident ratio, lower means less crowded)"""
    
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

@tool
def filter_by_public_transport(min_transport_score: int) -> Dict[str, Any]:
    """Filter cities by public transport quality (0-100, higher is better)"""
    
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
    """Filter cities by events/cultural scene quality and optionally by event types. Note: This requires subjective properties data."""
    
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

@tool
def filter_by_urban_nature(nature_preference: str) -> Dict[str, Any]:
    """Filter cities by urban nature and environment preference (pure_urban, urban_parks, nature_access, nature_immersed)"""
    
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

@tool
def filter_by_city_size(preferred_size: str) -> Dict[str, Any]:
    """Filter cities by size preference (intimate <500k, medium 500k-2M, metropolis >2M)"""
    
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
    """Filter cities by architectural style preference (historic, mixed, modern)"""
    
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

@tool
def filter_by_walkability(min_walkability_score: int) -> Dict[str, Any]:
    """Filter cities by walkability score (0-100, higher means more walkable and pedestrian-friendly)"""
    
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

@tool
def get_final_recommendations() -> Dict[str, Any]:
    """Get final ranked recommendations from current filtered cities"""
    
    state = search_states.get("current", SearchState())
    
    if len(state.current_cities) == 0:
        return {"error": "No cities match your criteria. Let's relax some constraints."}
    
    # Rank by composite score using available static properties
    ranked = state.current_cities.copy()
    ranked['composite_score'] = (
        (ranked['safety_score'] / 100) * 30 +        # Safety weight
        (ranked['healthcare_score'] / 100) * 25 +    # Healthcare weight  
        ((100 - ranked['cost_index']) / 100) * 20 +  # Cost efficiency weight (lower cost = higher score)
        (ranked['public_transport_score'] / 100) * 15 + # Transport weight
        (ranked['walkability_score'] / 100) * 10     # Walkability weight
    )
    
    top_cities = ranked.nlargest(min(5, len(ranked)), 'composite_score')
    
    recommendations = []
    for _, city in top_cities.iterrows():
        reasons = []
        if city['safety_score'] >= 85:
            reasons.append(f"Very safe (safety score: {city['safety_score']})")
        if city['healthcare_score'] >= 85:
            reasons.append(f"Excellent healthcare (score: {city['healthcare_score']})")
        if city['cost_index'] <= 60:
            reasons.append("Very affordable")
        elif city['cost_index'] <= 80:
            reasons.append("Good value")
        if city['public_transport_score'] >= 85:
            reasons.append("Excellent public transport")
        if city['walkability_score'] >= 85:
            reasons.append("Very walkable")
        
        recommendations.append({
            "city": city['city'],
            "country": city['country'],
            "score": round(city['composite_score'], 1),
            "reasons": reasons,
            "safety_score": city['safety_score'],
            "healthcare_score": city['healthcare_score'],
            "cost_index": city['cost_index'],
            "public_transport_score": city['public_transport_score'],
            "walkability_score": city['walkability_score'],
            "language_barrier": city['language_barrier'],
            "nature_access": city['nature_access'],
            "architectural_style": city['architectural_style']
        })
    
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

# Global search state storage (in production, use Redis)
search_states = {}

class ChatOrchestrator:
    def __init__(self):
        self.model_variant = "gpt-4"
        self.model = ChatOpenAI(
            temperature=0, 
            stream_usage=True, 
            streaming=True, 
            model=self.model_variant, 
            cache=False
        )
        
        # Bind all tools to the model
        tools = [
            filter_by_budget,
            filter_by_climate, 
            filter_by_safety,
            filter_by_language,
            filter_by_visa_requirements,
            filter_by_healthcare,
            filter_by_pollution,
            filter_by_tourism_load,
            filter_by_public_transport,
            filter_by_events,
            filter_by_urban_nature,
            filter_by_city_size,
            filter_by_architecture,
            filter_by_walkability,
            get_final_recommendations,
            analyze_event_preferences
        ]
        self.runnable_model = self.model.bind_tools(tools)
        
        self.sessions: Dict[str, Dict] = {}
        
        # System prompt focusing on information gain and progressive filtering
        self.system_prompt = """You are a travel recommendation agent that uses progressive filtering to find perfect destinations. You will ask questions to the user to gather information about their preferences and then use the filtering tools to progressively reduce where they can go.

Your approach:
1. Start with all available cities
2. Ask questions that provide maximum information gain to narrow down options
3. Use filtering tools to progressively reduce the search space
4. Continue asking questions until you have 7 or fewer cities remaining
5. Be conversational and lively - don't rigidly follow a script!

Available filtering tools:
- filter_by_budget(purpose, max_budget, budget_type, duration_days) - Essential first filter
- filter_by_climate(min_temp, max_temp, max_rainfall) - High impact filter (requires time-dependent data)
- filter_by_safety(min_safety_score) - Important for some users
- filter_by_language(max_language_barrier) - Important for some users
- filter_by_visa_requirements(passport_country, min_visa_free_days) - Important for long stays
- filter_by_healthcare(min_healthcare_score) - Important for some users
- filter_by_pollution(max_pollution_index) - Important for some users
- filter_by_tourism_load(max_tourism_ratio) - Important for some users
- filter_by_public_transport(min_transport_score) - Important for some users
- filter_by_events(min_events_score, event_types) - Helps with cultural fit (requires subjective data)
- filter_by_urban_nature(nature_preference) - Nature access preferences
- filter_by_city_size(preferred_size) - City size preferences
- filter_by_architecture(preferred_style) - Architectural style preferences
- filter_by_walkability(min_walkability_score) - Walkability requirements
- get_final_recommendations() - Call when you have 7 or fewer cities remaining

Suggested question priorities (adapt based on conversation flow):
1. PURPOSE & DURATION - "Quick break or months-long stay?" (short_stay vs long_stay - changes budget calculation)
2. BUDGET ANCHOR - Weekend: total trip cost (€) / Long-stay: monthly rent ceiling (€) (highest elimination rate)
   - Can filter by total budget, transport budget only, or daily accommodation cost
   - Ask clarifying questions: "Is that your total budget including flights?" or "Just for accommodation?"
3. CLIMATE PREFERENCE - Temperature range and rainfall tolerance (high impact filter, requires time-dependent data)
4. CITY CHARACTER - Size preference (intimate/medium/metropolis), nature access, architectural style
5. VISAS/PASSPORTS - (Long-stays only) Passport country and desired stay length 
6. SAFETY & HEALTHCARE - Safety importance and healthcare quality needs
7. LANGUAGE COMFORT - How important is English/local language comfort
8. POLLUTION - Air quality preferences 
9. TOURISM LOAD - Preference for authentic vs touristy destinations
10. PUBLIC TRANSPORT & WALKABILITY - Importance of transport and walkability
11. INTEREST HOOK - Event types/cultural scene preferences (requires subjective data)

Strategy:
- Ask questions that eliminate the most cities first (highest information gain)
- Adapt based on user responses - if they mention something, ask follow-up questions
- Keep it conversational - don't interrogate, have a natural chat
- After each filter, mention how many cities remain: "That narrows it down to X cities..."
- When you reach 7 or fewer cities, call get_final_recommendations()
- Be efficient but friendly - this is about finding the perfect match through conversation

Note: Some filters require additional data types:
- Climate filtering requires time-dependent properties (weather APIs)
- Events scoring requires subjective properties (calculated per user interests)
- Current dataset has static properties only, so some filters are placeholder implementations."""

    async def process_message_stream(self, client_id: str, message: dict) -> AsyncGenerator[dict, None]:
        """
        Process incoming message and stream response back
        """
        if client_id not in self.sessions:
            self.sessions[client_id] = {
                "messages": [],
            }
            # Initialize search state for this session
            search_states[client_id] = SearchState()
            search_states["current"] = search_states[client_id]  # Set as current for tools
        else:
            # Set this session's state as current
            search_states["current"] = search_states[client_id]
        
        session = self.sessions[client_id]
        
        # Add user message to session
        user_message = HumanMessage(content=message.get('content', ''))
        session["messages"].append(user_message)
        
        # Add current search state to context
        state_summary = search_states[client_id].get_state_summary()
        context_message = SystemMessage(content=f"""Current search state: {json.dumps(state_summary)}

{self.system_prompt}""")
        
        # Prepare all content for the model
        all_content = [
            context_message,
            *session["messages"]
        ]
        
        try:
            # Stream response from the model
            response_content = ""
            async for chunk in self.runnable_model.astream(all_content):
                if hasattr(chunk, 'content') and chunk.content:
                    response_content += chunk.content
                    yield {
                        "type": "stream",
                        "content": chunk.content,
                        "partial": True
                    }
                
                # Handle tool calls
                if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                    for tool_call in chunk.tool_calls:
                        yield {
                            "type": "tool_call",
                            "tool": tool_call['name'],
                            "args": tool_call.get('args', {}),
                            "status": "executing"
                        }

                        # Execute the tool call
                        try:
                            # Get tool function from globals
                            tool_func = globals().get(tool_call['name'])
                            if tool_func:
                                result = tool_func(**tool_call.get('args', {}))
                                yield {
                                    "type": "tool_result",
                                    "tool": tool_call['name'],
                                    "result": result,
                                    "status": "completed"
                                }
                            else:
                                yield {
                                    "type": "tool_result", 
                                    "tool": tool_call['name'],
                                    "result": {"error": f"Tool {tool_call['name']} not found"},
                                    "status": "failed"
                                }
                        except Exception as e:
                            logger.error(f"Tool execution error: {e}")
                            yield {
                                "type": "tool_result",
                                "tool": tool_call['name'], 
                                "result": {"error": str(e)},
                                "status": "failed"
                            }
            
            # Add assistant message to session
            if response_content.strip():
                session["messages"].append(AIMessage(content=response_content))
            
            # Send final response
            yield {
                "type": "message_complete",
                "content": response_content,
                "state": search_states[client_id].get_state_summary()
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            yield {
                "type": "error",
                "content": f"Error processing message: {str(e)}"
            } 