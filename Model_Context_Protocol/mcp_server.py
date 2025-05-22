from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient
import requests
from pprint import pprint
# Initialize FastMCP server
mcp = FastMCP("weather")

# Constants
NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"

async def make_nws_request(url: str) -> dict[str, Any] | None:
    """Make a request to the NWS API with proper error handling."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json"
    }

    #httpx.AsyncClient helps to process multiple concurrent requests. It is always used under a async function.
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

def format_alert(feature: dict) -> str:
    """Format an alert feature into a readable string."""
    props = feature["properties"]
    return f"""
Event: {props.get('event', 'Unknown')}
Area: {props.get('areaDesc', 'Unknown')}
Severity: {props.get('severity', 'Unknown')}
Description: {props.get('description', 'No description available')}
Instructions: {props.get('instruction', 'No specific instructions provided')}
"""

@mcp.tool()
async def get_alerts(state: str) -> str:
    """Get weather alerts for a US state.

    Args:
        state: Two-letter US state code (e.g. CA, NY)
    """
    url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    data = await make_nws_request(url)

    if not data or "features" not in data:
        return "Unable to fetch alerts or no alerts found."

    if not data["features"]:
        return "No active alerts for this state."

    alerts = [format_alert(feature) for feature in data["features"]]
    return "\n---\n".join(alerts)

@mcp.tool()
async def get_forecast(latitude: float, longitude: float) -> str:
    """Get weather forecast for a location.

    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
    """
    # First get the forecast grid endpoint
    points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"
    points_data = await make_nws_request(points_url)

    if not points_data:
        return "Unable to fetch forecast data for this location."

    # Get the forecast URL from the points response
    forecast_url = points_data["properties"]["forecast"]
    forecast_data = await make_nws_request(forecast_url)

    if not forecast_data:
        return "Unable to fetch detailed forecast."

    # Format the periods into a readable forecast
    periods = forecast_data["properties"]["periods"]
    forecasts = []
    for period in periods[:5]:  # Only show next 5 periods
        forecast = f"""
{period['name']}:
Temperature: {period['temperature']}°{period['temperatureUnit']}
Wind: {period['windSpeed']} {period['windDirection']}
Forecast: {period['detailedForecast']}
"""
        forecasts.append(forecast)

    return "\n---\n".join(forecasts)

@mcp.tool()
def search_external_info(query:str)->list:
    '''
    Search the internet for the given query and provides the list of search results.

    Args:
    query: The search phrase
    '''
    tavily_client = TavilyClient(api_key="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")
    search_results = tavily_client.search(query,max_results=2)
    return search_results

@mcp.tool()
async def nearest_place_finder_agent(prompt:str) -> list:
    '''
    This functions does a Google API map search and returns a list of top 5 queried places alongwith ratings
    and lat-long details.
    '''
    print('Fetching the list of nearest places!....')
    #st.pauls lat & lng
    curr_lat = 51.513446
    curr_lng = -0.099869
    headers = {
    "Content-Type": "application/json",
    "X-Goog-FieldMask": "places.displayName,places.location,places.id,places.formattedAddress,places.currentOpeningHours,places.priceLevel,places.rating",
    "X-Goog-Api-Key": "AIzaSyBFUhKxYJT_6mBKUF6ntcIdzPIivvi5-Jw"
    }

    data = {
    "textQuery" : f"{prompt}",
    "rankPreference": "DISTANCE",
    "maxResultCount": 5,
    "locationBias": {
    "circle": {
      "center": {
        "latitude": curr_lat,
        "longitude": curr_lng
        },
      "radius": 2000
        }
    }
    #"priceLevels":["PRICE_LEVEL_INEXPENSIVE","PRICE_LEVEL_MODERATE"]
    }

    response = requests.post(url="https://places.googleapis.com/v1/places:searchText",json=data,headers=headers)
    res = response.json()
    list_of_elements = []
    for elements in res['places']:
        dict_={"name":elements.get('displayName',{}).get('text'),
        "user_rating":elements.get('rating'),
        "latitude":elements.get('location',{}).get('latitude', ''),
        "longitude":elements.get('location', {}).get('longitude', ''),
        "Open_now":elements.get('currentOpeningHours', {}).get('openNow', '')
            }
        list_of_elements.append(dict_)
   
    return list_of_elements

@mcp.tool()
def navigation_agent(prompt:str) -> dict:
    '''
    Searches for the queried destination from the current location and opens the gmap navigation tool
    '''
    
    print('Opening the navigation tool for you!...')

    #st.paul's lat & lng
    curr_lat = 51.513446
    curr_lng = -0.099869
    headers = {
    "Content-Type": "application/json",
    "X-Goog-FieldMask": "places.displayName,places.id,places.formattedAddress,places.googleMapsLinks,places.location",
    "X-Goog-Api-Key": "AIzaSyBFUhKxYJT_6mBKUF6ntcIdzPIivvi5-Jw"
    }

    data = {
        "textQuery" : prompt,
        "rankPreference": "DISTANCE",
        "maxResultCount": 1,
        "locationBias": {
        "circle": {
        "center": {
            "latitude": curr_lat,
            "longitude": curr_lng
            },
        #"radius": 2000
            }
        }
    }

    response = requests.post(url="https://places.googleapis.com/v1/places:searchText",json=data,headers=headers)
    pprint(response.json())
    dest_lat = response.json()['places'][0]['location']['latitude']
    dest_lng = response.json()['places'][0]['location']['longitude']

    google_navigation_url = f"https://www.google.com/maps/dir/?api=1&origin={curr_lat},{curr_lng}&destination={dest_lat},{dest_lng}&travelmode=driving&key=AIzaSyBFUhKxYJT_6mBKUF6ntcIdzPIivvi5-Jw"
    
    # webbrowser.open_new(google_navigation_url)

    return {'url':google_navigation_url,'text':'I am now opening the navigation tool for you'}

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')