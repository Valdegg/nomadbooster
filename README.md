# Nomad Booster

## TL;DR

**Chat-first travel agent** that recommends where to travel/relocate by analyzing **14+ factors**: climate, cost, safety, visa rules, events, healthcare, etc. 

Uses **BrightData** to scrape protected sites (Skyscanner, Airbnb, Numbeo) behind bot-guards and CAPTCHAs. **Real-time WebSocket chat** with LangChain orchestrator progressively filters cities → personalized recommendations. 


Built for **BrightData's hackathon** — showcasing proxy rotation, browser automation, and unlocker tools for travel data that's impossible to scrape otherwise.

## Core Goals (Phase 1)

**Simple, working travel recommendation pipeline:**

🎯 **Static Filtering (2-3 filters)**
- Use **cost of living index** to filter by budget constraints
- Use **average temperature** for a given month to match climate preferences  
- Use **safety score** to ensure comfortable destinations

🚀 **Dynamic Filtering (1 filter on smaller set)**
- Use **flight prices** to customize recommendations based on actual travel costs
- Apply to the reduced city set from static filtering

🎉 **Find a Reason to Go**
- **Event matching**: Find cultural events/festivals that align with user interests
- Give users compelling reasons beyond just practical filters

🗺️ **Visual Results**
- **Show results on a map/globe** for intuitive destination exploration
- Make recommendations tangible and inspiring

*Focus: Get the basic pipeline working end-to-end before expanding to all 16 data sources*

---

A chat-first agent that decides where you should travel or relocate next by blending climate, cost, events, safety, healthcare, tax, language, transit quality and visa rules — then opens the flight page, Airbnb cart, or visa form for you.

## Project Structure (Monorepo)

```
├── api/                              # FastAPI + WebSocket backend
│   ├── main.py                       # WebSocket server with streaming
│   ├── orchestrator.py               # LangChain agent with tool binding (14 filters)
│   ├── models.py                     # Comprehensive Pydantic schemas
│   └── requirements.txt              # Dependencies
├── data/                             # Static datasets & data sources
│   ├── cities_static_properties.csv  # Static city properties (20 EU cities)
│   └── sources/                      # Data source schemas (16 JSON files)
├── datasource_integrations/          # Data fetching & processing scripts (16 files)
│   ├── numbeo_*.py                   # Cost & safety data integrations
│   ├── *_climate.py                  # Weather & climate APIs
│   ├── *_flight_costs.py             # Flight pricing scraping
│   ├── *_accommodation_costs.py      # Accommodation pricing automation
│   ├── *_events.py                   # Cultural events & scene analysis
│   └── *_visa.py                     # Visa requirements automation
├── scripts/                          # Aggregation & processing utilities
├── frontend/                         # React frontend (coming later)
└── docker-compose.yml
```

## Architecture

- **Real-time Chat**: WebSocket connections with streaming responses
- **LangChain Integration**: ChatOpenAI with 14 filter tools and comprehensive orchestration
- **Three-Tier Data**: Static, Time-Dependent, and Subjective property layers
- **16 Data Source Integrations**: Complete infrastructure for real-time data fetching
- **Progressive Filtering**: Intelligent questioning to narrow cities to optimal recommendations
- **Redis Cache**: TTL-based caching infrastructure ready

## How we use Bright Data  in Nomad Booster

*This project is built for BrightData's hackathon*

### Why we need it

Most of our key metrics—cost of living, visa rules, flight prices, Airbnb rates—sit behind JavaScript-heavy pages guarded by rate-limits or CAPTCHAs. Bright Data gives us:

1. **Residential & ISP proxy rotation** – requests look like real users, so Skyscanner, Airbnb and Numbeo don't block us.
2. **Web Unlocker + CAPTCHA solver** – auto-bypasses bot checks without extra code.
3. **Headless Browser API** – remote Chromium session we can drive with browser.open / click / type, ideal for interactive widgets like Timatic visa search.
4. **One-call "Structured scraper" tools** – site-specific extractors (e.g., Amazon, Zillow, TikTok) if we decide to expand later.
5. **Single SDK / tool name per call** – our agent just selects a tool; BrightData routes the request and returns parsed JSON or rendered HTML.

### Concretely, we leverage BrightData for:

| Metric we collect | Site | BrightData feature we call |
|-------------------|------|---------------------|
| **Flight cost** | Skyscanner "browse-quotes" page | `browser.open` (Unlocker) → scrape cheapest fare |
| **Accommodation cost** | Airbnb search results | `browser.open` + `browser.scroll` → scrape nightly median |
| **Cost of living & safety** | Numbeo (JS charts) | `browser.open` → extract COL & Crime indices |
| **Visa-free days & appointment backlog** | IATA Timatic widget + embassy booking form | `browser.open` → `browser.type` → `browser.click` → scrape table |
| *(optional) Event density* | Songkick city-events page | `scrape_as_markdown` (plain fetch wrapped by MCP) |

All other open datasets (Meteostat weather, Eurostat tourism load, EF English proficiency, etc.) are fetched directly via free CSV/JSON APIs—no proxy needed. Using BrightData only where it adds real value keeps run-time cost and complexity low while unlocking the hardest-to-scrape but most critical data.

## Development Tasks

### ✅ Task 1: Repo & DevOps scaffold (3h)
- [x] Monorepo folders (api/, data/, datasource_integrations/, scripts/)
- [x] Dockerfile & docker-compose.yml for API + Redis  
- [x] Basic configs (env.example, .gitignore)

### ✅ Task 2: Chat ↔ Orchestrator hook (6h)
- [x] Progressive filtering architecture (20 questions approach)
- [x] 14 filtering tools for comprehensive search space narrowing
- [x] Search state tracking (total cities, remaining cities, applied filters)
- [x] Information gain-based questioning strategy
- [x] LangChain ChatOpenAI with tool binding and streaming

### ✅ Task 3: Comprehensive Data Source Integration Infrastructure (12h)
- [x] **16 Data Source Integration Scripts** with full documentation
- [x] **Three-Tier Architecture**: Static, Time-Dependent, Subjective properties
- [x] **JSON Schema Documentation** for all data sources
- [x] **Complete Filter Coverage**: All user preferences covered by data sources
- [x] **Draft Implementation Structure**: Structured integration scripts ready for testing
- [x] **BrightData Integration Started**: Numbeo, Skyscanner, Airbnb scripts with BrightData implementation

### ✅ Task 4: Tool Calling Infrastructure & Core Filter Implementation (8h)
- [x] **Fixed LLM Tool Calling**: Resolved gpt-4 → gpt-4o model issues with proper tool calling support
- [x] **Streaming Tool Arguments**: Handled tool argument streaming and accumulation properly
- [x] **Parameter Extraction**: Natural language → structured parameters working ("between 15 and 25 degrees" → min_temp=15, max_temp=25)
- [x] **Climate Filter with Real Data**: Working filter using time-dependent sample data (20 cities → 9 cities filtering)
- [x] **Complete Conversation Flow**: Tool execution → LLM explanation → follow-up questions
- [x] **WebSocket Test Client**: Comprehensive testing infrastructure with timeout handling
- [x] **Search State Management**: Proper state tracking across conversation turns

### ✅ Task 5: Filter Implementation & Testing Completed (6h)
- [x] **All 10 Filter Tools Enabled**: Budget, climate, safety, language, healthcare, pollution, transport, nature, city size, walkability
- [x] **Filter Replacement Logic**: Same filter type replaces previous (no stacking), different types stack correctly
- [x] **Parameter Extraction Schemas**: All tools have LangChain-compatible schemas with required parameters and sentinel values
- [x] **Mock Data Compatibility**: All enabled filters work with static properties CSV
- [x] **Multiple Filter Combinations**: Tested climate + other filters successfully
- [x] **State Management**: Proper filter tracking and reapplication from scratch
- [x] **Natural Language Processing**: LLM correctly extracts filter parameters from conversational input

### ✅ Task 6: Frontend & User Experience Completed (4h)
- [x] **Simple HTML/CSS/JS Frontend**: Beautiful chat interface on port 3000
- [x] **Real-time WebSocket Integration**: Connects to API with streaming responses
- [x] **Tool Call Visualization**: Shows filter executions with blue boxes
- [x] **Search State Display**: Green boxes showing remaining cities and progress
- [x] **Auto-reconnection**: Handles connection drops gracefully
- [x] **Responsive Design**: Modern chat UI with typing indicators and status

### ✅ Task 7: Real Data Integration Implementation Completed (8h)
- [x] **Numbeo Cost & Safety Data**: Full BrightData integration for real cost items (meals, drinks, transport, rent) and safety scores
- [x] **Comprehensive Weather Data**: Open-Meteo climate API integration with 240 records (20 cities × 12 months)
- [x] **Enhanced Cost Filtering**: Granular item-based cost filtering (meal prices, taxi fares, rent costs) with closest-match fallbacks
- [x] **Advanced Climate Filtering**: Temperature ranges, rainfall, sunshine scores, UV index, precipitation probability, categorical weather preferences
- [x] **Real Data CSV Files**: `cities_cost_data.csv` and `cities_weather.csv` with complete datasets for all 20 European cities
- [x] **Geocoding Integration**: Automatic city coordinate resolution using Open-Meteo geocoding API
- [x] **Filter Enhancement**: Updated orchestrator to use real data instead of mock data for cost and weather filters

### ⏳ Task 8: Data Mapping & Normalization
- [ ] **Raw Data → Standardized Metrics**: Transform API responses to filter-ready format
- [ ] **Normalization Algorithms**: Scale different data sources to consistent 0-100 ranges
- [ ] **Data Validation**: Ensure data quality and handle missing values
- [ ] **Metric Mapping**: Map complex source data to simple filter parameters
- [ ] **Update Frequency Management**: Handle static vs dynamic data refresh cycles

### ⏳ Task 9: Additional Features & Polish
- [ ] **Final Recommendations Tool**: Test get_final_recommendations() when ≤7 cities remain
- [ ] **Edge Case Handling**: No results found, constraint relaxation
- [ ] **Advanced Frontend**: React with map visualization and scatter plots
- [ ] **Browser Automation**: Booking integration for flights/accommodation

## Quick Start

### 🌍 **Complete System (Recommended)**
```bash
# 1. Start the API backend
cd api
python main.py  # Runs on localhost:8000

# 2. Start the frontend (in new terminal)
cd frontend
python server.py  # Runs on localhost:3000, opens browser automatically
```

### Using Docker (API Only)
```bash
# Copy environment file
cp env.example .env
# Edit .env with your OpenAI API key: OPENAI_API_KEY=sk-...

# Start services
docker-compose up -d

# Check logs
docker-compose logs -f api
```

### Local Development (API Only)
```bash
# Backend setup
cd api
pip install -r requirements.txt
python main.py

# WebSocket endpoint: ws://localhost:8000/ws/{client_id}
# HTTP endpoint: http://localhost:8000
```

### 🎯 **What You Get:**
- **Backend API**: http://localhost:8000 with WebSocket chat
- **Frontend UI**: http://localhost:3000 with beautiful chat interface  
- **Real-time filtering**: Test all 10 filter combinations through conversation
- **Visual feedback**: See tool executions and search progress in real-time
- **Persistent sessions**: Conversations automatically saved and resumable

### 🔗 **Session Management:**
- **New conversation**: http://localhost:3000/ → automatically creates unique session URL
- **Resume conversation**: http://localhost:3000/chat_123456789_abc12 → resumes exact state
- **Share conversations**: Copy URL to share search state with others
- **Multiple sessions**: Each browser tab can have separate conversations
- **Auto-persistence**: All conversations, filters, and search progress saved to Redis

## Three-Tier Data Architecture

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
- `filter_by_budget()` - Flexible budget filtering (total/transport/accommodation)
- `filter_by_climate()` - Temperature and rainfall preferences
- `filter_by_safety()` - Safety requirements (0-100 scale)
- `filter_by_language()` - Language comfort (1-5 scale)
- `filter_by_visa_requirements()` - Visa policies for long stays
- `filter_by_healthcare()` - Healthcare quality requirements
- `filter_by_pollution()` - Air quality preferences

### Extended Filters:
- `filter_by_tourism_load()` - Tourist density vs authenticity
- `filter_by_public_transport()` - Transit quality needs
- `filter_by_events()` - Cultural scene and event preferences
- `filter_by_urban_nature()` - Green space accessibility
- `filter_by_city_size()` - Population size preferences
- `filter_by_architecture()` - Architectural style preferences
- `filter_by_walkability()` - Pedestrian-friendly environments

### Recommendation Tools:
- `get_final_recommendations()` - Ranked results when ≤7 cities remain

## Data Source Integration Status

### ✅ **Live Data Sources (4 sources)** - **Completed**
- **Numbeo Cost & Safety**: Real granular cost data (meals, drinks, rent, transport) + safety scores for all 20 cities
- **Open-Meteo Climate**: Comprehensive weather data (240 records: 20 cities × 12 months) with temperature, rainfall, sunshine, UV
- **Airbnb Accommodation**: Static accommodation costs (40 records: 20 cities × 2 property types) + live lookup tool for any city
- **Status**: ✅ **Fully implemented and live** - Real data CSV files generated and integrated into filtering system

### 📝 Metric Ready (7 remaining sources) - **Priority 1**
- **Static**: WHO/OECD healthcare, EF language, IATA visa, IQAir pollution, Moovit transport, UN DESA population  
- **Dynamic**: Skyscanner flights
- **Status**: Metrics are well-defined, draft scripts created, need testing and real API integration → **Implement next**
- **Challenge**: Need to test, debug, and implement normalization to 0-100 scales

### ⚠️ Algorithm Needed (3 sources) - **Priority 2** 
- **Eurostat tourism load**: Straightforward ratio calculation
- **OpenStreetMap nature**: Geometric area calculations  
- **Events scoring**: Algorithm design needed for 0-100 scoring
- **Status**: Requires custom algorithms to create the metrics → **Implement if time permits**
- **Challenge**: Custom algorithms for scoring/classification before data integration

### ❌ Complex Implementation (1 source) - **Future/Optional**
- **Wikipedia architecture**: Requires LLM analysis for style classification
- **Status**: Significant complexity → **Skip for MVP, consider later**

### **Current Coverage: 4 Live + 7-10 More Planned = 11-14 Total Data Sources**

## Data Mapping & Normalization Challenges

### 🔄 **Raw Data → Standardized Metrics**
**Challenge**: APIs return diverse data formats, need consistent filter inputs

**Examples**:
- **Numbeo Cost Index** (40-120) → `cost_index` (0-100 normalized)
- **Open-Meteo Temperature** (-20°C to 45°C) → `avg_temp_c` (validated range)
- **WHO Healthcare** (complex indicators) → `healthcare_score` (0-100 composite)
- **IATA Visa** (days/visa-type) → `visa_free_days` (integer, 0-365)

### 📊 **Normalization Algorithms Needed**
- **Min-Max Scaling**: Cost indices, pollution levels to 0-100 scales
- **Z-Score Normalization**: Population data for city size classification
- **Composite Scoring**: Multiple healthcare indicators → single score
- **Categorical Mapping**: Text descriptions → enum values (architectural styles)

### 🔍 **Data Quality & Validation**
- **Missing Data Handling**: Fallback values, interpolation strategies
- **Outlier Detection**: Validate data ranges, flag suspicious values
- **Freshness Tracking**: Cache invalidation based on data source update frequencies
- **Cross-Validation**: Sanity checks between related metrics

## Target Cities: MVP → Global Expansion

### 🎯 **Current State**: 20 European Cities with Real + Mock Data
Berlin, Amsterdam, Barcelona, Prague, Lisbon, Vienna, Rome, Paris, Copenhagen, Stockholm, Brussels, Madrid, Munich, Zurich, Dublin, Budapest, Warsaw, Athens, Helsinki, Oslo

**Status**: 
- ✅ **Real Data**: Cost items (Numbeo) and comprehensive weather data (Open-Meteo) - `cities_cost_data.csv` and `cities_weather.csv`
- 📋 **Mock Data**: Remaining properties in `cities_static_properties.csv` ready to be replaced with real data sources

### 🔄 **Next Step**: Complete Real Data Integration
- ✅ **2/11 Sources Live**: Numbeo cost/safety data and Open-Meteo climate data fully integrated
- ⏳ **Continue Implementation**: Add remaining 8 metric-ready sources (healthcare, language, visa, pollution, etc.)
- 🎯 **Goal**: Replace all mock data with real APIs - same city list and structure

### 🌍 **Scaling to 1000+ Cities**: Just Bigger City Lists
**Reality**: Once data fetching works for 20 cities, scaling is straightforward:
1. **Expand city list**: Add more cities to target lists in integration scripts
2. **Run same scripts**: Same data fetching code, just on more cities  
3. **Generate larger table**: Same CSV structure, more rows

**Regional Expansion**:
- **North America**: ~100 major US/Canadian cities
- **Asia-Pacific**: ~200 major cities across Asia, Australia, New Zealand  
- **Latin America**: ~100 major cities across Central/South America
- **Africa & Middle East**: ~100 major cities and capitals
- **Europe Extended**: ~100+ smaller European cities beyond the initial 20

**Key Insight**: The hard work is building the data fetching infrastructure. Once that works for 20 cities, scaling to 1000+ is just running the same programs on bigger lists.

## Progressive Filtering Strategy

### **Phase 1 Focus**: Implement Core Filtering with Metric-Ready Sources
**Target**: Get 11 📝 Metric-ready sources working and tested for solid recommendation foundation

1. **Purpose & Duration** → Budget calculation context
2. **Budget Anchor** → Highest information gain filter  
3. **Climate Preference** → Temperature and rainfall comfort (📝 Metric ready: Open-Meteo)
4. **Cultural Interests** → Events and cultural scene alignment (⚠️ Needs scoring algo)
5. **Administrative** → Visa requirements (📝 Metric ready: IATA data)
6. **Quality of Life** → Safety, healthcare, language comfort (📝 Metric ready: Numbeo, WHO/OECD, EF)
7. **Environmental** → Pollution (📝 Metric ready: IQAir) 
8. **Urban Style** → City size (📝 Metric ready: UN DESA), architecture (❌ Skip for MVP)
9. **Infrastructure** → Public transport quality (📝 Metric ready: Moovit)
10. **Market Pricing** → Flight/accommodation costs (📝 Metric ready + BrightData)

**Goal**: Test and implement 8-10 metric-ready sources, add algorithm-based ones incrementally.

## Development Phases

- **✅ Phase 0**: Complete agent infrastructure with comprehensive filtering + **Tool calling infrastructure working**
- **✅ Phase 1**: **Filter testing & implementation** - All 10 filter tools working with mock data, multi-filter combinations tested
- **✅ Phase 1.5**: **Frontend implementation** - Complete web interface with real-time chat and visualization
- **🔄 Phase 2**: **Real data integration** - ✅ 4 sources live (Numbeo cost/safety, Open-Meteo climate, Airbnb accommodation), 7 remaining metric-ready sources to implement
- **⏳ Phase 3**: **Data normalization & polish** - Map raw data to standardized metrics + advanced features  
- **⏳ Phase 4**: **Browser automation** - Add booking integration and advanced frontend features
- **🌍 Phase 5+**: **Global expansion** - Scale to 1000+ cities (just run scripts on bigger city lists)

**Current Status: Phase 1 & 1.5 Complete → Phase 2 (Real Data Integration) Ready**

**Recent Major Achievements**:
- ✅ **Real Data Integration**: Numbeo cost/safety data and Open-Meteo climate data (240 records) fully live!
- ✅ **Airbnb Accommodation Costs**: Static accommodation data (40 records: 20 cities × 2 property types) + live lookup tool for any city with BrightData
- ✅ **Enhanced Cost Filtering**: Granular accommodation cost filtering (entire places, private rooms, average) integrated into main cost filters
- ✅ **Enhanced Filtering**: Granular cost filtering (meals, drinks, rent) and advanced weather filtering (sunshine, UV, rainfall categories)
- ✅ **All 10 Filter Tools Working**: Budget, climate, safety, language, healthcare, pollution, transport, nature, city size, walkability
- ✅ **Filter replacement logic**: Same filter types replace instead of stack, different types combine correctly
- ✅ **Complete Frontend**: Beautiful web interface on localhost:3000 with real-time chat
- ✅ **Tool call visualization**: Shows filter executions, search progress, and remaining cities
- ✅ **Multi-filter conversations**: Test complex combinations with real cost/weather data + mock data for other factors

**Current Focus**: ✅ **Major Progress** - 4 key data sources live with real data! Continuing implementation of remaining 7 metric-ready sources.

**What You Can Test Now**:
- *"I want somewhere between 15-25 degrees in June, very safe, under €12 meals, good coffee under €3, private rooms under €100/night"*
- **Real data filtering** for temperature, weather conditions, meal prices, coffee costs, apartment rents, accommodation costs (entire places & private rooms), and safety scores!
- **Live accommodation lookup**: *"Look up accommodation in Frankfurt"* - get real-time Airbnb prices for any city
- **Advanced weather queries**: *"sunny weather with low UV"*, *"dry climate for hiking"*, *"bright sunshine in July"*
- All complex combinations work with mix of real data (cost/weather/accommodation) + mock data (other factors)

## API Endpoints

- `GET /` - API info & features
- `GET /health` - Health check + Redis status + orchestrator status
- `WS /ws/{client_id}` - WebSocket chat with streaming responses
- `GET /cities` - Static cities data

## WebSocket Message Format

**Client → Server:**
```json
{
  "content": "I want a weekend trip for under €500 with good nightlife",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

**Server → Client (Streaming):**
```json
{
  "type": "stream",
  "content": "I'd love to help you find the perfect destination...",
  "partial": true
}
```

```json
{
  "type": "tool_call", 
  "tool": "filter_by_budget",
  "status": "executing",
  "details": "Filtering by budget: €500 for weekend trip..."
}
```

```json
{
  "type": "stream_complete",
  "content": "Based on your preferences, I found 3 perfect matches...",
  "recommendations": [...]
}
```

## Target Users

- **Weekend Traveller**: "Where can I fly cheaply, enjoy mild weather and a good electronic music scene?"
- **Long-stay Digital Nomad**: "Which city has mild climate, low crime, decent healthcare and visa-free access for 6 months?"
- **Architecture Enthusiast**: "I love Art Nouveau buildings and need good public transport in a medium-sized city"

## Tech Stack

- **Backend**: Python with FastAPI + LangChain + WebSockets + Redis
- **AI**: ChatOpenAI with 14 filter tools and streaming responses
- **Data Processing**: Pydantic schemas + Pandas + comprehensive validation
- **Data Sources**: 16 integration scripts covering all travel factors
- **Scraping**: BrightData for dynamic pricing and protected sites
- **Frontend**: React with split-pane UI (chat + results visualization) [Coming]
- **Maps**: Integrated mapping for location visualization [Coming]