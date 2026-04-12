import random
from datetime import datetime


# Mock weather data — in production this would call a real weather API
MOCK_WEATHER_DATA = {
    "chennai": {"temp_c": 32, "condition": "Sunny", "humidity": 75, "wind_kph": 15},
    "mumbai": {"temp_c": 29, "condition": "Partly Cloudy", "humidity": 80, "wind_kph": 20},
    "bangalore": {"temp_c": 24, "condition": "Cloudy", "humidity": 65, "wind_kph": 10},
    "delhi": {"temp_c": 35, "condition": "Hazy", "humidity": 45, "wind_kph": 12},
    "coimbatore": {"temp_c": 28, "condition": "Clear", "humidity": 60, "wind_kph": 18},
    "hyderabad": {"temp_c": 31, "condition": "Sunny", "humidity": 55, "wind_kph": 14},
    "kolkata": {"temp_c": 30, "condition": "Humid", "humidity": 85, "wind_kph": 8},
    "london": {"temp_c": 14, "condition": "Rainy", "humidity": 90, "wind_kph": 25},
    "new york": {"temp_c": 18, "condition": "Clear", "humidity": 50, "wind_kph": 20},
    "tokyo": {"temp_c": 22, "condition": "Partly Cloudy", "humidity": 70, "wind_kph": 15},
}

# Tool schema — tells the LLM what this tool does and what parameters it needs
WEATHER_TOOL_SCHEMA = {
    "name": "get_weather",
    "description": "Get the current weather for a city. Use this when the user asks about weather, temperature, or climate conditions in any location.",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The name of the city to get weather for, e.g. Chennai, London, New York"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit. Default is celsius."
            }
        },
        "required": ["city"]
    }
}


def get_weather(city: str, unit: str = "celsius") -> dict:
    """
    Get mock weather data for a city.
    In production this would call OpenWeatherMap, WeatherAPI, etc.
    """
    city_lower = city.lower().strip()

    if city_lower in MOCK_WEATHER_DATA:
        data = MOCK_WEATHER_DATA[city_lower].copy()
    else:
        # Generate plausible random data for unknown cities
        data = {
            "temp_c": random.randint(15, 38),
            "condition": random.choice(["Sunny", "Cloudy", "Rainy", "Clear", "Windy"]),
            "humidity": random.randint(40, 90),
            "wind_kph": random.randint(5, 35)
        }

    temp_c = data["temp_c"]
    temp_f = round(temp_c * 9/5 + 32, 1)

    return {
        "city": city.title(),
        "temperature": temp_f if unit == "fahrenheit" else temp_c,
        "unit": "°F" if unit == "fahrenheit" else "°C",
        "condition": data["condition"],
        "humidity_percent": data["humidity"],
        "wind_kph": data["wind_kph"],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
    }