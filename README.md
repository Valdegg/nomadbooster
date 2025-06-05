# Nomad Booster

## TL;DR

**Chat-first travel agent** that recommends where to travel/relocate by analyzing **14+ factors**: climate, cost, safety, visa rules, events, healthcare, etc. Uses **BrightData MCP** to scrape protected sites (Skyscanner, Airbnb, Numbeo) behind bot-guards and CAPTCHAs. **Real-time WebSocket chat** with LangChain orchestrator progressively filters 20 EU cities → personalized recommendations. Built for **BrightData's hackathon** — showcasing proxy rotation, browser automation, and unlocker tools for travel data that's impossible to scrape otherwise.

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
- **Progressive Filtering**: Intelligent questioning to narrow 20 cities to optimal recommendations
- **Redis Cache**: TTL-based caching infrastructure ready

## How we use Bright Data MCP in Nomad Booster

*This project is built for BrightData's hackathon*

### Why we need it

Most of our key metrics—cost of living, visa rules, flight prices, Airbnb rates—sit behind JavaScript-heavy pages guarded by rate-limits or CAPTCHAs. Bright Data's Managed Control Plane (MCP) gives us:

1. **Residential & ISP proxy rotation** – requests look like real users, so Skyscanner, Airbnb and Numbeo don't block us.
2. **Web Unlocker + CAPTCHA solver** – auto-bypasses bot checks without extra code.
3. **Headless Browser API** – remote Chromium session we can drive with browser.open / click / type, ideal for interactive widgets like Timatic visa search.
4. **One-call "Structured scraper" tools** – site-specific extractors (e.g., Amazon, Zillow, TikTok) if we decide to expand later.
5. **Single SDK / tool name per call** – our agent just selects a tool; MCP routes the request and returns parsed JSON or rendered HTML.

### Concretely, we leverage MCP for:

| Metric we collect | Site | MCP feature we call |
|-------------------|------|---------------------|
| **Flight cost** | Skyscanner "browse-quotes" page | `browser.open` (Unlocker) → scrape cheapest fare |
| **Accommodation cost** | Airbnb search results | `browser.open` + `browser.scroll` → scrape nightly median |
| **Cost of living & safety** | Numbeo (JS charts) | `browser.open` → extract COL & Crime indices |
| **Visa-free days & appointment backlog** | IATA Timatic widget + embassy booking form | `browser.open` → `browser.type` → `browser.click` → scrape table |
| *(optional) Event density* | Songkick city-events page | `scrape_as_markdown` (plain fetch wrapped by MCP) |

All other open datasets (Meteostat weather, Eurostat tourism load, EF English proficiency, etc.) are fetched directly via free CSV/JSON APIs—no proxy needed. Using MCP only where it adds real value keeps run-time cost and complexity low while unlocking the hardest-to-scrape but most critical data.

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
- [x] **Ready for Implementation**: Structured integration ready for data fetching
- [x] **BrightData Integration Started**: Numbeo, Skyscanner, Airbnb scripts with MCP implementation

### ⏳ Task 4: Data Fetching Implementation
- [ ] **Priority 1**: Implement easy wins (11 ✅ Ready sources)
- [x] **BrightData MCP Integration**: Started for Numbeo (cost/safety), Skyscanner (flights), Airbnb (accommodation)
- [ ] **Priority 2**: Simple calculations (3 ◻️ sources) as time permits
- [ ] Redis cache decorator setup (30d/1h/real-time TTL)
- [ ] Data aggregation and processing pipeline
- [ ] Error handling and fallback mechanisms

### ⏳ Task 5: Data Mapping & Normalization
- [ ] **Raw Data → Standardized Metrics**: Transform API responses to filter-ready format
- [ ] **Normalization Algorithms**: Scale different data sources to consistent 0-100 ranges
- [ ] **Data Validation**: Ensure data quality and handle missing values
- [ ] **Metric Mapping**: Map complex source data to simple filter parameters
- [ ] **Update Frequency Management**: Handle static vs dynamic data refresh cycles

### ⏳ Task 6: Frontend & User Experience
- [ ] React frontend with chat interface
- [ ] Results visualization (map + scatter plots)
- [ ] Browser automation for booking integration

## Quick Start

### Using Docker (Recommended)
```bash
# Copy environment file
cp env.example .env
# Edit .env with your OpenAI API key: OPENAI_API_KEY=sk-...

# Start services
docker-compose up -d

# Check logs
docker-compose logs -f api
```

### Local Development
```bash
# Backend setup
cd api
pip install -r requirements.txt
python main.py

# WebSocket endpoint: ws://localhost:8000/ws/{client_id}
# HTTP endpoint: http://localhost:8000
```

## Three-Tier Data Architecture

### 🏛️ Static Properties (Annual Updates)
**Properties inherent to cities that rarely change**

| Data Source | Metric | Status | Integration |
|-------------|--------|---------|-------------|
| **Numbeo Cost Index** | `cost_index` | ✅ Ready | `numbeo_cost_index.py` |
| **Numbeo Safety Score** | `safety_score` | ✅ Ready | `numbeo_safety_score.py` |
| **WHO/OECD Healthcare** | `healthcare_score` | ✅ Ready | `who_oecd_healthcare.py` |
| **EF Language Barrier** | `language_barrier` | ✅ Ready | `ef_language_barrier.py` |
| **IATA Visa Requirements** | `visa_free_days` | ✅ Ready | `iata_timatic_visa.py` |
| **IQAir Pollution Index** | `pollution_index` | ✅ Ready | `iqair_pollution_index.py` |
| **Moovit Transport Score** | `public_transport_score` | ✅ Ready | `moovit_public_transport.py` |
| **UN DESA Population** | `city_size` | ✅ Ready | `undesa_city_population.py` |
| **OpenStreetMap Nature** | `nature_access` | ◻️ Compute ratio | `openstreetmap_nature_access.py` |
| **Wikipedia Architecture** | `architectural_style` | ❌ LLM needed | `wikipedia_architectural_style.py` |

### ⏰ Time-Dependent Properties (Real-time/Daily)
**Properties that vary based on travel dates**

| Data Source | Metric | Status | Integration |
|-------------|--------|---------|-------------|
| **Meteostat/Open-Meteo** | `avg_temp_c`, `rainfall_mm` | ✅ Ready | `meteostat_openmeteo_climate.py` |
| **Skyscanner Flight Costs** | `flight_cost_eur` | ✅ Ready | `skyscanner_flight_costs.py` |
| **Airbnb Accommodation** | `accommodation_cost_eur` | ✅ Ready | `airbnb_accommodation_costs.py` |
| **Eurostat Tourism Load** | `tourism_load_ratio` | ◻️ Simple formula | `eurostat_tourism_load.py` |

### 🎯 Subjective Properties (User-specific)
**Properties calculated based on user preferences**

| Data Source | Metric | Status | Integration |
|-------------|--------|---------|-------------|
| **Songkick/Bandsintown** | `events_score`, `cultural_alignment` | ◻️ Scoring needed | `songkick_bandsintown_events.py` |

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

### ✅ Implementation Ready (11 sources) - **Priority 1**
- **Static**: Numbeo (cost, safety), WHO/OECD healthcare, EF language, IATA visa, IQAir pollution, Moovit transport, UN DESA population
- **Dynamic**: Meteostat/Open-Meteo climate, Skyscanner flights, Airbnb accommodation
- **Status**: Well-documented APIs, clear integration paths → **Implement first**
- **Challenge**: Need normalization to 0-100 scales or similar for consistent filtering

### ◻️ Simple Implementation (3 sources) - **Priority 2** 
- **Eurostat tourism load**: Straightforward ratio calculation
- **OpenStreetMap nature**: Geometric area calculations  
- **Events scoring**: Algorithm design needed for 0-100 scoring
- **Status**: Doable with some effort → **Implement if time permits**
- **Challenge**: Custom algorithms for scoring/classification

### ❌ Complex Implementation (1 source) - **Future/Optional**
- **Wikipedia architecture**: Requires LLM analysis for style classification
- **Status**: Significant complexity → **Skip for MVP, consider later**

### **Realistic Coverage: 11-14 Data Sources** (focus on easy wins + some simple ones)

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

### 🎯 **Current State**: 20 European Cities with Mock Data
Berlin, Amsterdam, Barcelona, Prague, Lisbon, Vienna, Rome, Paris, Copenhagen, Stockholm, Brussels, Madrid, Munich, Zurich, Dublin, Budapest, Warsaw, Athens, Helsinki, Oslo

**Status**: Static mockup data in `cities_static_properties.csv` for testing filters

### 🔄 **Next Step**: Replace Mock Data with Real Data  
- Run data source integration scripts on the same 20 cities
- Populate real `cities_static_properties.csv` with actual data from APIs
- Same city list, same table structure, just real data instead of mock values

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

### **Phase 1 Focus**: Implement Core Filtering with Easy Data Sources
**Target**: Get 11 ✅ Ready sources working for solid recommendation foundation

1. **Purpose & Duration** → Budget calculation context
2. **Budget Anchor** → Highest information gain filter  
3. **Climate Preference** → Temperature and rainfall comfort (✅ Easy: Open-Meteo)
4. **Cultural Interests** → Events and cultural scene alignment (◻️ Skip for MVP)
5. **Administrative** → Visa requirements (✅ Easy: IATA data)
6. **Quality of Life** → Safety, healthcare, language comfort (✅ Easy: Numbeo, WHO/OECD, EF)
7. **Environmental** → Pollution (✅ Easy: IQAir) 
8. **Urban Style** → City size (✅ Easy: UN DESA), architecture (❌ Skip for MVP)
9. **Infrastructure** → Public transport quality (✅ Easy: Moovit)
10. **Market Pricing** → Flight/accommodation costs (✅ Easy: APIs available)

**Goal**: Deliver solid recommendations with 8-10 working filters, add more incrementally.

## Development Phases

- **✅ Phase 0**: Complete agent infrastructure with comprehensive filtering
- **⏳ Phase 1**: **Core data sources** - Implement 11 ✅ Ready sources, replace mock data with real data
- **⏳ Phase 2**: **Data normalization** + Frontend - Map raw data to standardized metrics + UI development  
- **⏳ Phase 3**: **Polish & Features** - Add remaining sources, browser automation
- **🌍 Phase 4+**: **Global expansion** - Scale to 1000+ cities (just run scripts on bigger city lists)

**Current Status: Phase 0 Complete → Ready for Phase 1 implementation**

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
- **Scraping**: Selenium/Playwright + MCP for dynamic pricing
- **Frontend**: React with split-pane UI (chat + results visualization) [Coming]
- **Maps**: Integrated mapping for location visualization [Coming]