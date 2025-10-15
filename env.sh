export NSIDC_WMS_LAYER="NSIDC:seaice_conc_daily_north"  # update to your exact WMS layer if needed
export NSIDC_WMS_BASE="https://nsidc.org/api/mapservices?service=WMS&request=GetMap"
export NSIDC_WMS_LAYER="NSIDC:seaice_conc_daily_north"

export NSIDC_DATA_DIR="$PWD/data"
export NSIDC_OPENDAP_URL="$NSIC_DATA_DIR/sic_psn25_20250915_F17_icdr_v03r00.nc"
export NSIDC_VAR_NAME="cdr_seaice_conc"
export NSIDC_CRS="EPSG:3411"

export NSIDC_SENSORS="F18,F17"
export NSIDC_URL_PATTERN="https://noaadata.apps.nsidc.org/NOAA/G10016_V3/CDR/north/daily/{yyyy}/sic_psn25_{yyyymmdd}_{sensor}_icdr_v03r00.nc"

export AIS_EXPORTVESSELS_KEY=""
export AIS_EXPORTVESSELTRACK_KEY=""
export AIS_SHIPSEARCH_KEY=""
export AIS_VESSELPHOTO_KEY=""
export AIS_PORTCALLS_KEY=""
export AIS_VESSELEVENTS_KEY=""