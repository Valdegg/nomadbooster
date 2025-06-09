

# 1. get which cities we have 
# read cities from data/european_iatas_df.csv
# 2. 
# fetch numbeo data for them 
# create_cities_csv.py to make _real.csv 
# airbnb_accommodation_costs.py to make _with_accommodations.csv 
# 3. get weather data for them
# meteostat_openmeteo_climate.py  for weather data
# convert_climate_to_csv.py to get cities_weather.csv  

import pandas as pd

# 1. get which cities we have 
# read cities from data/sources/european_iatas_df.csv
european_iatas_df = pd.read_csv('european_iatas_df.csv')

# 2. fetch numbeo data for them 
# create_cities_csv.py to make _real.csv 
# airbnb_accommodation_costs.py to make _with_accommodations.csv 