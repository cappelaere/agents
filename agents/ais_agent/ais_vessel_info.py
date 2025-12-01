"""
Helpers for querying vessel ownership and particulars via GraphQL.

Author: Patrice G. Cappelaere, IBM Federal

This module wraps the Kpler/MarineTraffic GraphQL endpoint to fetch vessel
information by IMO, MMSI, or owner/management name. It is used by the AIS
agent to enrich vessel responses with detailed ownership and particulars.
"""

import requests
import json, os
import logging

logger = logging.getLogger("ais_vessel_info")
if not logger.handlers:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# - 1. Set API endpoint and authentication
BASE_URL    = "https://api.kpler.marinetraffic.com/v2/vessels/graphql"
API_KEY     = os.getenv("AIS_OWNERSHIP_KEY", "")


def _escape_graphql_string(value: str) -> str:
    """Escape a Python string for safe inclusion in a GraphQL string literal."""

    return value.replace("\\", "\\\\").replace('"', '\\"')


def _post_graphql(query: str) -> dict:
    """Post a GraphQL query to the Kpler/MarineTraffic endpoint with safe error handling."""

    headers = {
        "Authorization": f"Basic {API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            BASE_URL,
            json={"query": query},
            headers=headers,
            timeout=30.0,
        )
    except requests.RequestException as exc:
        logger.info("AIS GraphQL request exception: %s", exc)
        logger.error(
            "AIS GraphQL request failed: url=%s error=%s",
            BASE_URL,
            exc,
            exc_info=True,
        )
        return {
            "error": "UpstreamServiceUnavailable",
            "service": "ais_graphql",
            "message": "Vessel info service is unreachable. See server logs for details.",
        }

    if response.status_code != 200:
        logger.info("AIS GraphQL upstream error: status=%s", response.status_code)
        logger.error(
            "AIS GraphQL upstream error: status=%s body=%r",
            response.status_code,
            response.text[:2000],
        )
        return {
            "error": "UpstreamServiceError",
            "service": "ais_graphql",
            "status": response.status_code,
            "message": "Vessel info service returned an error. See server logs for details.",
        }

    try:
        data = response.json()
    except ValueError as exc:
        logger.info("AIS GraphQL JSON decode error: %s", exc)
        logger.error(
            "AIS GraphQL JSON decode error: %s body=%r",
            exc,
            response.text[:1000],
        )
        return {
            "error": "UpstreamServiceError",
            "service": "ais_graphql",
            "status": response.status_code,
            "message": "Vessel info service returned invalid JSON. See server logs for details.",
        }
    return data

def fetch_vessel_info_by_imo(imo, after_cursor=None):
    """Fetch detailed vessel information from Kpler/MarineTraffic by IMO.

    Args:
        imo: IMO vessel identifier.
        after_cursor: Optional pagination cursor for subsequent pages.

    Returns:
        dict | None: Parsed JSON response from the GraphQL API, or ``None`` on error.
    """
    # - 3. Define GraphQL query: you can comment out some of the sections to include more fields in the response.
    imo_safe = _escape_graphql_string(str(imo))
    query = f"""
    query Vessels {{
        vessels(
            first: 100  # Number of records per page, max value is 1000
            where: {{
                filters: [
                    {{
                        field: "identifier.imo"
                        op: IN
                        values: ["{imo_safe}"]
                    }}
                    #{{
                    #    field: "identifier.mmsi"
                    #    op: EQ
                    #    values: ["9172571"]
                    #}}
                    #{{
                    #    field: "management.beneficialOwner.current.name"
                    #    op: LIKE
                    #    values: ["AASEN SHIPPING%"]
                    #}}
                ]
                operator: OR   # To combine multiple filters, use OR or AND
            }}
            after: {json.dumps(after_cursor)}  # Dynamically add cursor for pagination
        ) {{
            nodes {{

                ### Identifier - (Un)comment fields below to include/exclude them from the response
                identifier {{
                    imo
                    mmsi
                    callSign
                    eni
                    shipId
                }}

                ### Management - (Un)comment fields below to include/exclude them from the response
                management {{
                    beneficialOwner {{ current {{ name country address website startDate }} }}
                    registeredOwner {{ current {{ name country address website startDate }} }}
                    commercialManager {{ current {{ name country address website startDate }} }}
                    operator {{ current {{ name country address website startDate }} }}
                    technicalManager {{ current {{ name country address website startDate }} }}
                    ismManager {{ current {{ name country address website startDate }} }}
                }}

                ### Associated Companies - (Un)comment fields below to include/exclude them from the response
                # associatedCompanies {{
                #     shipBuilder {{ current {{ name country address website startDate }} }}
                #     engineBuilder {{ current {{ name country address website startDate }} }}
                #     classificationSociety {{ current {{ name country address website startDate }} }}
                #     piClub {{ current {{ name country address website startDate }} }}
                # }}

                ### Vessel Particulars - (Un)comment fields below to include/exclude them from the response
                particulars {{
                    general {{
                        name
                        commercialFleet
                        generalVesselType
                        detailedVesselType
                        serviceStatus
                        flag
                        portOfRegistry
                        keelLaidDate
                    }}
                    # hull {{
                    #     yearOfBuild
                    #     yardNumber
                    #     hullMaterial
                    #     hullType
                    #     decks
                    # }}
                    # aisTransceiver {{
                    #     lengthFore
                    #     lengthAft
                    #     widthLeft
                    #     widthRight
                    #     aisTransceiverClass
                    # }}
                    dimension {{
                        lengthOverall
                        lengthBetweenPerpendiculars
                        breadthExtreme
                        breadthMoulded
                        draught
                        depth
                        freeboard
                    }}
                    tonnage {{
                        grossTonnage
                        deadweightTonnage
                        netTonnage
                        loadedDisplacementTonnage
                        lightDisplacementTonnage
                    }}
                    # capacity {{
                    #     liquidCapacity
                    #     gasCapacity
                    #     baleCapacity
                    #     grainCapacity
                    #     teuCapacity
                    #     ceuCapacity
                    #     passengerCapacity
                    #     ballastCapacity
                    # }}
                    # engine {{
                    #     enginePower
                    #     engineUnits
                    #     engineCylinderUnits
                    #     engineBore
                    #     engineStroke
                    #     engineRpm
                    #     engineType
                    #     speedService
                    #     propeller
                    # }}
                    # fuel {{
                    #     mainEngineFuelType
                    #     fuelCapacity
                    # }}
                }}
            }}

            # Pagination Info
            pageInfo {{
                hasNextPage
                endCursor
            }}
        }}
    }}
    """

    # - 4–7. Send the request with safe error handling
    return _post_graphql(query)

# =====================================================
def fetch_vessel_info_by_mmsi(mmsi, after_cursor=None):
    """Fetch detailed vessel information from Kpler/MarineTraffic by MMSI.

    Args:
        mmsi: Maritime Mobile Service Identity.
        after_cursor: Optional pagination cursor for subsequent pages.

    Returns:
        dict | None: Parsed JSON response from the GraphQL API, or ``None`` on error.
    """
    # - 3. Define GraphQL query: you can comment out some of the sections to include more fields in the response.
    mmsi_safe = _escape_graphql_string(str(mmsi))
    logger.info("Fetching vessel info for MMSI: %s", mmsi_safe)
    query = f"""
    query Vessels {{
        vessels(
            first: 100  # Number of records per page, max value is 1000
            where: {{
                filters: [
                    #{{
                    #    field: "identifier.imo"
                    #    op: IN
                    #    values: [""]
                    #}}
                    {{
                        field: "identifier.mmsi"
                        op: EQ
                        values: ["{mmsi_safe}"]
                    }}
                    #{{
                    #    field: "management.beneficialOwner.current.name"
                    #    op: LIKE
                    #    values: ["AASEN SHIPPING%"]
                    #}}
                ]
                operator: OR   # To combine multiple filters, use OR or AND
            }}
            after: {json.dumps(after_cursor)}  # Dynamically add cursor for pagination
        ) {{
            nodes {{

                ### Identifier - (Un)comment fields below to include/exclude them from the response
                identifier {{
                    imo
                    mmsi
                    callSign
                    eni
                    shipId
                }}

                ### Management - (Un)comment fields below to include/exclude them from the response
                management {{
                    beneficialOwner {{ current {{ name country address website startDate }} }}
                    registeredOwner {{ current {{ name country address website startDate }} }}
                    commercialManager {{ current {{ name country address website startDate }} }}
                    operator {{ current {{ name country address website startDate }} }}
                    technicalManager {{ current {{ name country address website startDate }} }}
                    ismManager {{ current {{ name country address website startDate }} }}
                }}

                ### Associated Companies - (Un)comment fields below to include/exclude them from the response
                # associatedCompanies {{
                #     shipBuilder {{ current {{ name country address website startDate }} }}
                #     engineBuilder {{ current {{ name country address website startDate }} }}
                #     classificationSociety {{ current {{ name country address website startDate }} }}
                #     piClub {{ current {{ name country address website startDate }} }}
                # }}

                ### Vessel Particulars - (Un)comment fields below to include/exclude them from the response
                particulars {{
                    general {{
                        name
                        commercialFleet
                        generalVesselType
                        detailedVesselType
                        serviceStatus
                        flag
                        portOfRegistry
                        keelLaidDate
                    }}
                    # hull {{
                    #     yearOfBuild
                    #     yardNumber
                    #     hullMaterial
                    #     hullType
                    #     decks
                    # }}
                    # aisTransceiver {{
                    #     lengthFore
                    #     lengthAft
                    #     widthLeft
                    #     widthRight
                    #     aisTransceiverClass
                    # }}
                    dimension {{
                        lengthOverall
                        lengthBetweenPerpendiculars
                        breadthExtreme
                        breadthMoulded
                        draught
                        depth
                        freeboard
                    }}
                    tonnage {{
                        grossTonnage
                        deadweightTonnage
                        netTonnage
                        loadedDisplacementTonnage
                        lightDisplacementTonnage
                    }}
                    # capacity {{
                    #     liquidCapacity
                    #     gasCapacity
                    #     baleCapacity
                    #     grainCapacity
                    #     teuCapacity
                    #     ceuCapacity
                    #     passengerCapacity
                    #     ballastCapacity
                    # }}
                    # engine {{
                    #     enginePower
                    #     engineUnits
                    #     engineCylinderUnits
                    #     engineBore
                    #     engineStroke
                    #     engineRpm
                    #     engineType
                    #     speedService
                    #     propeller
                    # }}
                    # fuel {{
                    #     mainEngineFuelType
                    #     fuelCapacity
                    # }}
                }}
            }}

            # Pagination Info
            pageInfo {{
                hasNextPage
                endCursor
            }}
        }}
    }}
    """
    # - 4–7. Send the request with safe error handling
    return _post_graphql(query)

# =====================================================
def fetch_vessel_info_by_name(name, after_cursor=None):
    """Fetch detailed vessel information from Kpler/MarineTraffic by owner name.

    Args:
        name: Owner or management name fragment to search (used in a LIKE filter).
        after_cursor: Optional pagination cursor for subsequent pages.

    Returns:
        dict | None: Parsed JSON response from the GraphQL API, or ``None`` on error.
    """
    # - 3. Define GraphQL query: you can comment out some of the sections to include more fields in the response.
    name_safe = _escape_graphql_string(str(name))
    query = f"""
    query Vessels {{
        vessels(
            first: 100  # Number of records per page, max value is 1000
            where: {{
                filters: [
                    #{{
                    #    field: "identifier.imo"
                    #    op: IN
                    #    values: [""]
                    #}}
                    #{{
                    #    field: "identifier.mmsi"
                    #   op: EQ
                    #    values: [""]
                    #}}
                    {{
                        field: "management.beneficialOwner.current.name"
                        op: LIKE
                        values: ["{name_safe}"]
                    }}
                ]
                operator: OR   # To combine multiple filters, use OR or AND
            }}
            after: {json.dumps(after_cursor)}  # Dynamically add cursor for pagination
        ) {{
            nodes {{

                ### Identifier - (Un)comment fields below to include/exclude them from the response
                identifier {{
                    imo
                    mmsi
                    callSign
                    eni
                    shipId
                }}

                ### Management - (Un)comment fields below to include/exclude them from the response
                management {{
                    beneficialOwner {{ current {{ name country address website startDate }} }}
                    registeredOwner {{ current {{ name country address website startDate }} }}
                    commercialManager {{ current {{ name country address website startDate }} }}
                    operator {{ current {{ name country address website startDate }} }}
                    technicalManager {{ current {{ name country address website startDate }} }}
                    ismManager {{ current {{ name country address website startDate }} }}
                }}

                ### Associated Companies - (Un)comment fields below to include/exclude them from the response
                # associatedCompanies {{
                #     shipBuilder {{ current {{ name country address website startDate }} }}
                #     engineBuilder {{ current {{ name country address website startDate }} }}
                #     classificationSociety {{ current {{ name country address website startDate }} }}
                #     piClub {{ current {{ name country address website startDate }} }}
                # }}

                ### Vessel Particulars - (Un)comment fields below to include/exclude them from the response
                particulars {{
                    general {{
                        name
                        commercialFleet
                        generalVesselType
                        detailedVesselType
                        serviceStatus
                        flag
                        portOfRegistry
                        keelLaidDate
                    }}
                    # hull {{
                    #     yearOfBuild
                    #     yardNumber
                    #     hullMaterial
                    #     hullType
                    #     decks
                    # }}
                    # aisTransceiver {{
                    #     lengthFore
                    #     lengthAft
                    #     widthLeft
                    #     widthRight
                    #     aisTransceiverClass
                    # }}
                    dimension {{
                        lengthOverall
                        lengthBetweenPerpendiculars
                        breadthExtreme
                        breadthMoulded
                        draught
                        depth
                        freeboard
                    }}
                    tonnage {{
                        grossTonnage
                        deadweightTonnage
                        netTonnage
                        loadedDisplacementTonnage
                        lightDisplacementTonnage
                    }}
                    # capacity {{
                    #     liquidCapacity
                    #     gasCapacity
                    #     baleCapacity
                    #     grainCapacity
                    #     teuCapacity
                    #     ceuCapacity
                    #     passengerCapacity
                    #     ballastCapacity
                    # }}
                    # engine {{
                    #     enginePower
                    #     engineUnits
                    #     engineCylinderUnits
                    #     engineBore
                    #     engineStroke
                    #     engineRpm
                    #     engineType
                    #     speedService
                    #     propeller
                    # }}
                    # fuel {{
                    #     mainEngineFuelType
                    #     fuelCapacity
                    # }}
                }}
            }}

            # Pagination Info
            pageInfo {{
                hasNextPage
                endCursor
            }}
        }}
    }}
    """

    # - 4–7. Send the request with safe error handling
    return _post_graphql(query)

# - 8. Fetch first page
#imo = 9411410
#data = fetch_vessel_info_by_imo(imo)

# - 9. Display formatted JSON output
#if data:
#    print("Success! API Response:")
#    print(json.dumps(data, indent=2))  # Pretty-print JSON

    # - 10. Handle pagination (automatically fetch additional pages)
    #while data["data"]["vessels"]["pageInfo"]["hasNextPage"]:
    #    next_cursor = data["data"]["vessels"]["pageInfo"]["endCursor"]
    #    print("\n Fetching next page...")
    #    data = fetch_vessels(after_cursor=next_cursor)
    #    print(json.dumps(data, indent=2))  # Pretty-print subsequent pages