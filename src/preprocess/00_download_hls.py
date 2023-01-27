
# https://nasa-openscapes.github.io/2021-Cloud-Hackathon/tutorials/05_Data_Access_Direct_S3.html
# Instructions for getting access to the NASA data: https://urs.earthdata.nasa.gov/documentation/for_users/data_access/curl_and_wget

# imports
from datetime import datetime, timedelta
import os
import requests
import boto3
import numpy as np
import xarray as xr
import rasterio as rio
from rasterio.session import AWSSession

import rioxarray
import pandas as pd
import json
import rasterio
import rasterio.mask
from loguru import logger


START_IDX = 0 # index starts from 0
END_IDX = 2  # None means to finish until end (index also starts at 0)
FAIL_LOG_FP = "failure_indices.txt"
DATA_DIR = "../data"                            # where data is stored (csv, json reference files)
DATE_STR_FMT = '%Y-%m-%d'
OUT_DIR = "/Volumes/R Sandisk SSD/hls_tmp"    # folder where data will be downloaded to

ssc_json_path = f'{DATA_DIR}/ssc_sample_2573.json'
aqsat_path = f'{DATA_DIR}/Aqusat_TSS_v1.csv'
aqsat_path = f'{DATA_DIR}/L08_data_exactloc.csv'
s3_cred_endpoint = 'https://data.lpdaac.earthdatacloud.nasa.gov/s3credentials'

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

def get_temp_creds():
    temp_creds_url = s3_cred_endpoint
    return requests.get(temp_creds_url).json()

def read_obs(obs, json_data, exact_date=False, entry=999, lc_idx=None):
    """NOTE: only one of exact_date and approx_date should be True"""

    lat, lon = str(obs['lat']), str(obs['long'])
    dict_key = str(lon) +','+ str(lat)
    logger.debug(f"Getting data for dict_key={dict_key}, date={obs['date']}")

    if exact_date:
        date = obs['date']
    elif entry!=999:
        date = list(json_data[dict_key].keys())[entry]
    else:
        logger.debug('Please select a date or entry')
        return

    try:
        hls_names = json_data[dict_key][date]
    except KeyError:
        logger.error(f"KeyError: {dict_key} not found in `json_data`")
        with open(FAIL_LOG_FP, "a") as fp:
            fp.write(f"[{dict_key}, {date}]: {lc_idx} not found in `json_data`,\n")
        return [] # return no data

    return hls_names


logger.debug("Loading data from reference files")

# data import
aqsat = pd.read_csv(aqsat_path)
with open(ssc_json_path, 'r') as f:
    data = json.load(f)


logger.debug("Getting temp credentials for downloading HLS data")
temp_creds_req = get_temp_creds()

# load boto
session = boto3.Session(
    aws_access_key_id=temp_creds_req['accessKeyId'], 
    aws_secret_access_key=temp_creds_req['secretAccessKey'],
    aws_session_token=temp_creds_req['sessionToken'],
    region_name='us-west-2'
)

rio_env = rio.Env(AWSSession(session),
    GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
    GDAL_HTTP_COOKIEFILE=os.path.expanduser('~/cookies.txt'),
    GDAL_HTTP_COOKIEJAR=os.path.expanduser('~/cookies.txt')
)

rio_env.__enter__()

for lc_idx in range(len(aqsat)):
    if lc_idx < START_IDX:
        continue
    if END_IDX and lc_idx > END_IDX:
        continue
    logger.info(f"Landsat data {lc_idx:03d}/{len(aqsat)-1}")
    obs = aqsat.iloc[lc_idx]

    lat, long = obs["lat"], obs["long"]
    key = str(long) +','+ str(lat)

    # Get all dates, +/-1 day from listed day (including original date)
    date_orig = obs['date']
    date_obj = datetime.strptime(date_orig, DATE_STR_FMT).date()
    dates = [
        datetime.strftime(date_obj - timedelta(x), DATE_STR_FMT)
        for x in [-1,0,1]
    ]
    
    for date in dates:
        if not os.path.exists(os.path.join(OUT_DIR, key)):          # make folder with lat,long as name
            os.makedirs(os.path.join(OUT_DIR, key))
        if not os.path.exists(os.path.join(OUT_DIR, key, date)):    # make folder with date as name
            os.makedirs(os.path.join(OUT_DIR, key, date))

        hls_links = read_obs(obs, data, exact_date=True, lc_idx=lc_idx)
        
        datetime_filter = None
        tile_id_filter = None
        for idx, url in enumerate(sorted(hls_links)):
            # TODO: use time from aqsat to filter the links (i.e., get data closest to aqsat `date_utc`)
            # TODO: need to add time in json file if needed (since there can be multiple tiles in a day)
            logger.debug(f"Getting data from: {url}")

            # send a HTTP request and save
            fp_out = os.path.join(OUT_DIR, key, date, url.split("/")[-1])
            r = requests.get(url) # create HTTP response object
            with open(fp_out,'wb') as f:
                f.write(r.content)

            # rx_data = rioxarray.open_rasterio(url, chuncks=True, masked=False).squeeze()
            # # rx_data = rasterio.open(url, chuncks=True, masked=True, parse_coordinates=True)

            # # Write to TIFF and save
            # fp_out = os.path.join(OUT_DIR, key, date, url.split("/")[-1])
            # logger.debug(f"Saving band to file: {fp_out}")
            # try:
            #     rx_data.rio.to_raster(fp_out)
            # except rasterio.errors.RasterioIOError:
            #     logger.error(f"Error saving file to {fp_out}")
            #     with open(FAIL_LOG_FP, "a") as fp:
            #         fp.write(f"RIO ERROR [{lat},{long}, {date}]: {lc_idx},\n")

        