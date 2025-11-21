import requests
import json, os

# - 1. Set API endpoint and authentication
BASE_URL    = "https://api.kpler.marinetraffic.com/v2/vessels/graphql"
API_KEY     = os.getenv("AIS_OWNERSHIP_KEY", "")

# - 2. Fetch vessel data with pagination
def fetch_vessel_info_by_imo(imo, after_cursor=None):
    query = f"""
    query Vessels {{
        vessels(
            first: 100
            where: {{
                filters: [
                    {{
                        field: "identifier.imo"
                        op: IN
                        values: ["{imo}"]
                    }}
                ]
                operator: OR
            }}
            after: {json.dumps(after_cursor)}
        ) {{
            nodes {{ identifier {{ imo mmsi callSign eni shipId }} }}
            pageInfo {{ hasNextPage endCursor }}
        }}
    }}
    """

    headers = {
        "Authorization": f"Basic {API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(BASE_URL, json={"query": query}, headers=headers)
    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        return None
    return response.json()


def fetch_vessel_info_by_mmsi(mmsi, after_cursor=None):
    return fetch_vessel_info_by_imo("", after_cursor)


def fetch_vessel_info_by_name(name, after_cursor=None):
    return fetch_vessel_info_by_imo("", after_cursor)
