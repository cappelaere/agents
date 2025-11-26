import os, pathlib
import json, time
import logging
logger = logging.getLogger()   
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logger.setLevel(LOG_LEVEL)


def post_geojson(geojson):
    """Post the geojson to the ingest endpoint."""
    import requests
    featureCount = len(geojson.get('features', []))
    logger.info(f"Posting geojson with {featureCount} features")
    logger.debug
    HOST = os.getenv("HOST", "localhost")
    INGEST_URL = f"http://{HOST}:8120/ingest"
    headers = {
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(INGEST_URL, headers=headers, json=geojson)
        response.raise_for_status()
        logger.info(f"Successfully posted geojson to {INGEST_URL}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error posting geojson to {INGEST_URL}: {e}")

def make_geojson(payload, ship_track= True):
    """Store the payload to a file for debugging or auditing purposes."""
    logger.info(f"ship_track_geojson {payload}")

    # get each element of payload array and write to a file
    features = []
    coordinates = []   
    for element in payload:        
        # Store the coordinates for ship track
        coordinates.append( [element['LON'], element['LAT']] )
        # Create the feature
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [ element['LON'], element['LAT'] ]
            },
            "properties": {
                'MMSI': element['MMSI'],
                'IMO': element['IMO'],
                "SHIP_ID": element['SHIP_ID'],
                'STATUS': element['STATUS'],
                'SPEED': element['SPEED'],
                'COURSE': element['COURSE'],
                'HEADING': element['HEADING'],
                'TIMESTAMP': element['TIMESTAMP']   
            }
        }
        features.append(feature)
    
    if ship_track:
        # Creates the last feature which is the ship track
        track_feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coordinates
            },
            "properties": {
                "description": "Ship Track"
            }
        }    

        features.append(track_feature)
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    post_geojson(geojson)

def ships_geojson(payload):
    """Transform an array of ships to geojson and post it to the map."""
    logger.info(f"ships_geojson {payload}")
    make_geojson(payload, ship_track= False)
   
def ship_track_geojson(payload):
    """Transform an array of ship positions to geojson and post it to the map with the ship track."""
    logger.info(f"ship_track_geojson {payload}")
    make_geojson(payload, ship_track= True)



