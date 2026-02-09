# Historical Flight Data

## Data Provenance

Document the source of all historical data to ensure reproducibility and proper attribution.

### Schiphol Airport (EHAM)

**Source**: [To be filled when data is obtained]

**Potential Sources**:
- OpenSky Network (https://opensky-network.org/) - Free ADS-B data
- FlightRadar24 API (requires subscription)
- Schiphol Developer Portal (https://developer.schiphol.nl/)
- Aviation Edge API
- Manual data collection from flight tracking websites

**License**: [Specify data license]

**Date Range**: [When available]

**Preprocessing Applied**:
- [ ] Aircraft type standardization (map ICAO codes to categories)
- [ ] Time zone conversion to local time
- [ ] Filtering (removed cargo-only, private, military flights?)
- [ ] Runway mapping (map runway names to indices)

**Known Issues**:
- [ ] Missing data for certain time periods
- [ ] Aircraft type incomplete for some flights
- [ ] Runway assignment may be inferred, not actual

---

## How to Obtain Schiphol Data

### Option 1: OpenSky Network (Free)
```python
# Example: Query OpenSky Historical Database
from traffic.data import opensky

# Get arrivals to EHAM on 2024-01-15
flights = opensky.history(
    start="2024-01-15 00:00",
    stop="2024-01-15 23:59",
    arrival_airport="EHAM"
)
```

### Option 2: Schiphol API (Registration Required)
```bash
# Register at https://developer.schiphol.nl/
# API provides real-time and historical flight data
curl -X GET "https://api.schiphol.nl/public-flights/flights" \
  -H "resourceversion: v4" \
  -H "app_id: YOUR_APP_ID" \
  -H "app_key: YOUR_APP_KEY"
```

### Option 3: FlightRadar24 (Paid)
Commercial API access available with historical data.

---

## Data Schema Mapping

When converting from external sources to our format:

| Our Field | OpenSky | Schiphol API | FlightRadar24 |
|-----------|---------|--------------|---------------|
| `flight_id` | `callsign` | `flightName` | `flight` |
| `scheduled_time` | `firstSeen` | `scheduleTime` | `std` |
| `actual_time` | `lastSeen` | `actualLandingTime` | `atd` |
| `aircraft_type` | `typecode` | `aircraftType.iataMain` | `model` |
| `runway` | *inferred* | *not provided* | *not provided* |

**Note**: Runway assignments often need to be inferred from trajectory data or are not publicly available.