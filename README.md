# Nomad Booster

## TL;DR

**AI-powered travel recommendation system** that analyzes **14+ factors** (climate, cost, safety, visa requirements, healthcare, etc.) to suggest optimal travel destinations. Features real-time WebSocket chat with progressive filtering and comprehensive data integration.

Uses **BrightData** for scraping protected travel sites (Skyscanner, Airbnb, Numbeo) with proxy rotation and CAPTCHA solving.

## Recent Updates ✨

### 🗺️ **City Coordinates & Data Display**
- **City coordinates integration** from real geolocation data
- **Three-table data display**: Cost, Weather, and Other data organized separately 
- **Proper currency symbols** based on each city's local currency (€, kr, $, £, etc.)
- **Conditional weather data** only shown when travel month is specified

### 🔧 **Token Management & Performance**
- **Token overflow resolution** with `verbose` parameter to prevent context length issues
- **Token usage logging** for input/output monitoring and cost tracking
- **LLM filtering reminders** for progressive search refinement
- **Redis session persistence** with full state restoration

### 💾 **Enhanced Data Integration**
- **Real climate data** from Open-Meteo for all European cities (240 records)
- **Comprehensive cost data** from Numbeo with granular item filtering
- **Dynamic accommodation lookup** with live Airbnb-style pricing
- **Flight price integration** ready for Skyscanner data

## Architecture

- **Real-time Chat**: WebSocket with streaming LLM responses and tool execution
- **Progressive Filtering**: Intelligent questioning to narrow 210+ cities to optimal recommendations
- **Three-Tier Data**: Static properties, time-dependent data, and subjective preferences
- **Redis Caching**: Session persistence and data caching with TTL
- **BrightData Integration**: Proxy rotation for protected travel sites

## Quick Start

### 🚀 **Complete System**
```bash
# 1. Backend API
cd api
python main.py  # localhost:8000

# 2. Frontend UI  
cd frontend
python server.py  # localhost:3000 (auto-opens browser)
```

### 🔧 **Environment Setup**
```bash
cp env.example .env
# Add your OpenAI API key: OPENAI_API_KEY=sk-...
```

### 🎯 **What You Get**
- **Conversational filtering**: "20-25 degrees in August" → temperature + month filters
- **Real-time data**: Live cost, weather, and accommodation information
- **Visual feedback**: Tool executions and search progress in chat
- **Session persistence**: Resumable conversations with full state
- **Currency accuracy**: Proper symbols and pricing for each destination

## Data Sources & Integration

### ✅ **Implemented**
| Source | Data | Status |
|--------|------|--------|
| **Open-Meteo** | Weather, climate data | ✅ Live (240 records) |
| **Numbeo** | Cost of living, safety scores | ✅ Real data with BrightData |
| **Airbnb** | Accommodation pricing | ✅ Static + live lookup |
| **City Coordinates** | Geolocation mapping | ✅ Complete coverage |
| **Dohop** | Flight prices | BrightData scripts ready |

### 🔄 **Ready for Integration**
| Source | Data | Integration |
|--------|------|-------------|
| **IATA Timatic** | Visa requirements | Browser automation ready |
| **WHO/OECD** | Healthcare quality | API endpoints documented |
| **EF Education** | Language barriers | Data schemas prepared |

## Key Features

### 🏛️ Static Properties (Annual Updates)
**Properties inherent to cities that rarely change**

| Data Source | Metric | Implementation | Integration |
|-------------|--------|---------|-------------|
| **Numbeo Cost Items** | `meal_inexpensive`, `meal_midrange_2p`, `cappuccino`, `beer_domestic`, `taxi_1mile`, `apartment_1br_outside`, `apartment_1br_center` | ✅ **Implemented** (real pricing data) | `numbeo_cost_data.py` |
| **Numbeo Safety Score** | `safety_score` | ✅ **Implemented** (real safety data) | `numbeo_safety_score.py` |
| **WHO/OECD Healthcare** | `healthcare_score` | 📝 Draft (metric ready) | `who_oecd_healthcare.py` |
| **EF Language Barrier** | `language_barrier` | 📝 Draft (metric ready) | `ef_language_barrier.py` |
| **IATA Visa Requirements** | `visa_free_days` | 📝 Draft (metric ready) | `iata_timatic_visa.py` |
| **IQAir Pollution Index** | `pollution_index` | 📝 Draft (metric ready) | `iqair_pollution_index.py` |
| **Moovit Transport Score** | `public_transport_score` | 📝 Draft (metric ready) | `moovit_public_transport.py` |
| **UN DESA Population** | `city_size` | 📝 Draft (metric ready) | `undesa_city_population.py` |
| **OpenStreetMap Nature** | `nature_access` | ⚠️ Draft (needs algo) | `openstreetmap_nature_access.py` |
| **Wikipedia Architecture** | `architectural_style` | ❌ Complex (LLM needed) | `wikipedia_architectural_style.py` |

### ⏰ Time-Dependent Properties (Real-time/Daily)
**Properties that vary based on travel dates**

| Data Source | Metric | Implementation | Integration |
|-------------|--------|---------|-------------|
| **Open-Meteo Climate** | `temp_min_c`, `temp_max_c`, `rainfall_mm`, `sunshine_score`, `uv_index_max` | ✅ **Implemented** (240 records live) | `meteostat_openmeteo_climate.py` |
| **Skyscanner Flight Costs** | `flight_cost_eur` | 📝 Draft (metric ready) + BrightData | `skyscanner_flight_costs.py` |
| **Airbnb Accommodation** | `accommodation_cost_eur` | ✅ **Implemented** (static data + live lookup) + BrightData | `airbnb_accommodation_costs.py` |
| **Eurostat Tourism Load** | `tourism_load_ratio` | ⚠️ Draft (needs algo) | `eurostat_tourism_load.py` |

### 🎯 Subjective Properties (User-specific)
**Properties calculated based on user preferences**

| Data Source | Metric | Implementation | Integration |
|-------------|--------|---------|-------------|
| **Songkick/Bandsintown** | `events_score`, `cultural_alignment` | ⚠️ Draft (needs scoring algo) | `songkick_bandsintown_events.py` |

## Comprehensive Filter Tools (14 Total)

The orchestrator now includes **14 filter tools** covering all user preferences:

### Core Filters:
- `filter_by_costs()` - Flexible budget filtering (/transport/accommodation)
- `filter_by_climate()` - Temperature and rainfall preferences
- `filter_by_safety()` - Safety requirements (0-100 scale)

### Extended Filters:
- `filter_by_language()` - Language comfort (1-5 scale)
- `filter_by_visa_requirements()` - Visa policies for long stays
- `filter_by_healthcare()` - Healthcare quality requirements
- `filter_by_pollution()` - Air quality preferences

- `filter_by_tourism_load()` - Tourist density vs authenticity
- `filter_by_public_transport()` - Transit quality needs
- `filter_by_events()` - Cultural scene and event preferences
- `filter_by_urban_nature()` - Green space accessibility
- `filter_by_city_size()` - Population size preferences
- `filter_by_architecture()` - Architectural style preferences
- `filter_by_walkability()` - Pedestrian-friendly environments

### Non-filter tools
- Search for flights, live
- Search for events 

## Development Status

### ✅ **Completed**
- ✅ Chat interface with real-time WebSocket streaming
- ✅ 10+ filter tools with natural language processing  
- ✅ Real weather and cost data integration
- ✅ Session persistence and state management
- ✅ BrightData integration for protected sites
- ✅ Token usage monitoring and overflow prevention
- ✅ Currency display and coordinate mapping
- ✅ Three-table data organization
- ✅ Flight price integration (Dohop via BrightData)

### 🔄 **In Progress**  
- 🔄 Additional data sources (healthcare, language barriers)
- 🔄 Map visualization for results
- Graphs for e.g. flight cost and accommodation per day (how long can you stay)
- 🔄 Advanced accommodation lookup features
- Explore city (with picture an wiki link)

### 📝 Metric Ready (7 remaining sources) - **Priority 1**
- **Static**: WHO/OECD healthcare, EF language, IATA visa, IQAir pollution, Moovit transport, UN DESA population  
- **Status**: Metrics are well-defined, draft scripts created, need testing and real API integration → **Implement next**
- **Challenge**: Need to test, debug, and implement normalization to 0-100 scales

## Project Structure

```
├── api/                    # FastAPI + WebSocket backend
│   ├── main.py            # WebSocket server
│   ├── orchestrator.py    # LangChain agent with 14 filters
│   └── requirements.txt   
├── data/                   # Datasets and mappings
│   ├── european_iatas_df.csv
│   ├── city_coordinates_mapping.csv
│   └── cities_weather.csv
├── datasource_integrations/ # Data fetching scripts
│   ├── numbeo_*.py        # Cost and safety data
│   ├── *_climate.py       # Weather APIs
│   └── *_accommodation_costs.py
├── frontend/              # HTML/CSS/JS interface
│   ├── index.html         # Chat interface
│   └── server.py          # Static file server
└── docker-compose.yml
```

Built for **BrightData's hackathon** — demonstrating proxy rotation, browser automation, and data extraction from bot-protected travel sites.
