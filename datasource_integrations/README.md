# Data Source Integrations

This directory contains Python scripts for fetching and processing data from external sources to populate our static city properties dataset.

## Directory Structure

```
datasource_integrations/          # Python integration scripts
├── numbeo_cost_index.py          # Numbeo cost-of-living index
├── numbeo_safety_score.py        # Numbeo crime → safety score
├── who_oecd_healthcare.py        # WHO + OECD healthcare metrics
├── ef_language_barrier.py        # EF English Proficiency → language barrier
├── iata_timatic_visa.py          # IATA Timatic visa requirements
└── iqair_pollution_index.py      # IQAir + WHO air quality

data/sources/                      # JSON output files
├── numbeo_cost_index.json         # Cost index data
├── numbeo_safety_score.json       # Safety score data  
├── who_oecd_healthcare.json       # Healthcare score data
├── ef_language_barrier.json       # Language barrier data
├── iata_timatic_visa.json         # Visa requirements data
└── iqair_pollution_index.json     # Pollution index data
```

## Data Source Mapping

| Metric | Source | Access Method | Status |
|--------|--------|---------------|---------|
| `cost_index` | Numbeo Cost-of-Living | HTML scraping | ✅ Ready |
| `safety_score` | Numbeo Crime Index | HTML scraping | ✅ Ready |
| `healthcare_score` | WHO UHC + OECD | CSV/API | ✅ Ready |
| `language_barrier` | EF English Proficiency | CSV | ✅ Ready |
| `visa_free_days` | IATA Timatic | Playwright automation | ✅ Ready |
| `pollution_index` | IQAir + WHO PM2.5 | JSON/CSV | ✅ Ready |

## Integration Workflow

1. **Data Fetching**: Each script fetches raw data from external sources
2. **Processing**: Convert/normalize data to standardized schemas
3. **City Mapping**: Map country-level data to individual cities
4. **Output**: Save processed data to JSON files in `data/sources/`
5. **Aggregation**: Combine all sources into final `cities_static_properties.csv`

## Implementation Notes

- All scripts are currently stub implementations with detailed docstrings
- Each script includes schema definitions and sample data structures
- Error handling and rate limiting should be implemented for production use
- Consider implementing incremental updates to avoid re-fetching all data

## BrightData Setup

The Numbeo integrations use BrightData's Browser API to bypass anti-bot protections:

### 1. Get BrightData Credentials
1. Sign up at [BrightData](https://brightdata.com)
2. Create a new zone for web scraping
3. Note your credentials: `customer-zone-username:password`

### 2. Set Environment Variable
```bash
export BRIGHTDATA_ENDPOINT="wss://brd-customer-hl_0dc2f720-zone-scraping_browser1:q1psjos7szad@brd.superproxy.io:9222"
```

### 3. Install Playwright
```bash
pip install playwright
playwright install chromium
```

### 4. Test the Integration
```bash
cd datasource_integrations
python test_numbeo_fetcher.py
```

## Usage

```bash
# Test Numbeo BrightData integration
python datasource_integrations/test_numbeo_fetcher.py

# Run individual data source integration
python datasource_integrations/numbeo_cost_index.py

# Or run all integrations (when implemented)
python scripts/aggregate_static_data.py
```

## Dependencies

Required Python packages:
- `requests` - HTTP requests
- `beautifulsoup4` - HTML parsing
- `pandas` - Data manipulation
- `playwright` - Browser automation with BrightData
- `logging` - Logging utilities

## Next Steps

1. Implement actual data fetching logic in each script
2. Add error handling and retry mechanisms
3. Create aggregation script to combine all sources
4. Set up automated data refresh pipeline
5. Add data validation and quality checks 