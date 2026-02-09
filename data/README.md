# Data Directory

This directory contains **input data** for simulations and experiments.

## Directory Structure

### `schedules/`
Flight arrival schedules used as inputs to the MDP.

- **`synthetic/`**: Programmatically generated schedules for testing and controlled experiments
- **`historical/`**: Real-world data from actual airport operations

### `topologies/`
Physical airport layouts (gate positions, runway configurations, taxiways).

---

## Data Sources

### Synthetic Schedules
Generated using `src/mdp/components/schedule_generator.py` with various arrival patterns:
- `uniform_*.json`: Evenly distributed arrivals
- `morning_rush_*.json`: Normal peak around 7:00 AM
- `evening_peak_*.json`: Normal peak around 7:00 PM
- `disrupted_*.json`: Schedules with simulated runway closures or delays

### Historical Schedules

#### Schiphol (EHAM)
**Source**: [Specify data source when obtained, e.g., "Schiphol API", "ADS-B Exchange", "FlightRadar24"]

**Date Range**: [When available]

**Format**: JSON with fields:
- `flight_id`: Unique identifier
- `scheduled_time`: Scheduled arrival (minutes from midnight)
- `actual_time`: Actual arrival time (if available)
- `runway`: Runway used for landing
- `aircraft_type`: Specific aircraft model (e.g., "B737-800")
- `aircraft_category`: Mapped category (e.g., "narrow-body")
- `airline`, `origin`, `registration`: Metadata

**Data Quality Notes**:
- [Any known issues, gaps, or preprocessing applied]

---

## File Formats

### JSON Schedule Format
```json
{
  "metadata": {
    "scenario_name": "...",
    "date": "YYYY-MM-DD",
    "airport": "IATA/ICAO code",
    "num_flights": 123,
    "source": "..."
  },
  "flights": [
    {
      "flight_id": "...",
      "scheduled_time": 365,
      "runway": 0,
      "aircraft_type": "narrow-body",
      ...
    }
  ]
}
```

### CSV Schedule Format
```csv
flight_id,scheduled_time,actual_time,runway,aircraft_type,aircraft_category,...
KLM1234,365,368,0,B737-800,narrow-body,...
```

---

## Git LFS / .gitignore

Large historical datasets (>10MB) should be tracked with Git LFS or excluded from version control.

Current `.gitignore` rules:
```
# Large historical data
data/schedules/historical/**/*.json
data/schedules/historical/**/*.csv

# Keep synthetic data and README
!data/schedules/synthetic/
!data/README.md
```

---

## Adding New Data

### Synthetic Schedules
1. Run `python scripts/generate_schedule.py --pattern <pattern> --flights <N>`
2. Output saved to `data/schedules/synthetic/`

### Historical Data
1. Place raw data in `data/schedules/historical/<airport>/`
2. Document source in this README
3. Run preprocessing script if needed: `python scripts/preprocess_historical.py`
4. Update `.gitignore` if files are large

---

## Data Catalog

| File | Flights | Time Window | Pattern | Source | Notes |
|------|---------|-------------|---------|--------|-------|
| `synthetic/morning_rush_80.json` | 80 | 6AM-10AM | Normal peak @ 7AM | Generated | seed=42 |
| `synthetic/uniform_50.json` | 50 | 8-hour window | Uniform | Generated | seed=123 |
| `historical/schiphol/2024_01_15.json` | 324 | Full day | Actual | [TBD] | - |