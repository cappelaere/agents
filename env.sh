

export NSIDC_WMS_LAYER="NSIDC:seaice_conc_daily_north"  # update to your exact WMS layer if needed
export NSIDC_WMS_BASE="https://nsidc.org/api/mapservices?service=WMS&request=GetMap"
export NSIDC_WMS_LAYER="NSIDC:seaice_conc_daily_north"

export NSIDC_DATA_DIR="$PWD/data"
export NSIDC_OPENDAP_URL="$NSIC_DATA_DIR/sic_psn25_20250915_F17_icdr_v03r00.nc"
export NSIDC_VAR_NAME="cdr_seaice_conc"
export NSIDC_CRS="EPSG:3411"

export NSIDC_SENSORS="F18,F17"
export NSIDC_URL_PATTERN="https://noaadata.apps.nsidc.org/NOAA/G10016_V3/CDR/north/daily/{yyyy}/sic_psn25_{yyyymmdd}_{sensor}_icdr_v03r00.nc"

export AIS_EXPORTVESSELS_KEY="fd74c58e32b15115a3d78b467cc6877c8f9746b2"
export AIS_SHIPSEARCH_KEY="51e96a2ba97c149fc66f6a7afa96c1800a560869"
export AIS_VESSELPHOTO_KEY="62f4586b4f7005b410e9734e8a7aa5edab2ad69d"
export AIS_PORTCALLS_KEY="3a06b272b24a976ee9bd4c874443572a4c4f2b3e"
export AIS_VESSELEVENTS_KEY="d5794deb0dd5cc5364def3afcc905373b22c494b"

