# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 21:29:13 2024

@author: Debabrata Ghorai, Ph.D.

Generate shoreline transects.
This project only handle shapefile and geojson file.

"""

import os
# import sys
# import time
# import streamlit as st

# from tqdm.notebook import tqdm  # Specialized for Jupyter/VS Code
from osgeo import ogr
from src.utils.ref_scripts import fixed_interval_points
from src.geo_apps.vector_ops.find_nearest_point import get_nearest_point
from src.utils.geo_utils import (
    convert_to_shapefile, 
    check_vector_file_format,
    unlink_shapefiles
)
# from src.utils.progress_bar import pbar_object, update_pbar


# # Function to check environment
# def is_streamlit():
#     # Streamlit defines certain internal modules when running
#     return 'streamlit' in sys.modules and any('streamlit' in arg for arg in sys.argv)


# # Import the correct progress bar
# def pbar_object():
#     if is_streamlit():
#         import streamlit as st
#         # CSS to hide text on webpage
#         st.markdown("""<style>
#             .stProgress [data-testid="stWidgetLabel"] {display: none;}
#             .stProgress div[role="progressbar"] > div > div {color: transparent;}
#         </style>""", unsafe_allow_code=True)

#         pbar = st.progress(0)
#     else:
#         from tqdm.notebook import tqdm  # Specialized for Jupyter/VS Code
#         pbar = tqdm(total=100, desc="Executing Tasks")
#     return pbar


# # Update logic that works for both
# def update_pbar(pbar, val, increment):
#     pbar.progress(round(val / 100, 2)) if is_streamlit() else pbar.update(increment)
#     return


def create_shoreline_transects(onshore_line=None, offshore_line=None, out_transect_line=None, x_interval=None, progress_callback=None):
    # Define pbar in one line
    # pbar = pbar_object()
    val = 0

    # with stqdm(total=100, desc="Executing Tasks") as pbar:
    # task=1
    # check file format
    file_format1 = check_vector_file_format(onshore_line)
    onshore_line = convert_to_shapefile(
        src_file=onshore_line, 
        dest_file=onshore_line.replace(".geojson", ".shp")) if file_format1 == "GeoJSON" else onshore_line
    
    file_format2 = check_vector_file_format(offshore_line)
    offshore_line = convert_to_shapefile(
        src_file=offshore_line, 
        dest_file=offshore_line.replace(".geojson", ".shp")) if file_format2 == "GeoJSON" else offshore_line
    # update progressbar
    val += 1
    # progress_callback(val, 1)
    progress_callback(1)
    
    # task-2
    # read shapefile
    driver = ogr.GetDriverByName("ESRI Shapefile")
    ds_base = driver.Open(onshore_line, 0)
    lyr = ds_base.GetLayer(0)
    sr = lyr.GetSpatialRef()
    sr.ExportToWkt()
    ds_xline = driver.CreateDataSource(out_transect_line)
    lyr = ds_xline.CreateLayer('myLyr', sr, ogr.wkbMultiLineString)
    lyrDef = lyr.GetLayerDefn()
    # update progressbar
    val += 1
    # progress_callback(val, 1)
    progress_callback(1)
    
    # task-3
    # generate interval points
    baseline_points = fixed_interval_points(
        onshore_line, x_interval, flipline=True)
    # update progressbar
    val += 1
    # progress_callback(val, 1)
    progress_callback(1)
    
    # task-4
    if len(baseline_points) > 0:
        pbar_cnt = round((100-5)/len(baseline_points), 2)
        # generate cross-sections
        for ix, pnt in enumerate(baseline_points):
            # get nearest point
            near_pnt = get_nearest_point(pnt, offshore_line)
            # get the forward node (projected node)
            ix_f = near_pnt[0]+(near_pnt[0]-pnt[0])
            iy_f = near_pnt[1]+(near_pnt[1]-pnt[1])
            fnd = [ix_f, iy_f]
            # transect drawing
            x_line = ogr.Geometry(ogr.wkbLineString)
            x_line.AddPoint(pnt[0], pnt[1])
            x_line.AddPoint(fnd[0], fnd[1])
            # create feature
            feature = ogr.Feature(lyrDef)
            feature.SetGeometry(x_line)
            feature.SetFID(ix+1)
            lyr.CreateFeature(feature)
            # update progressbar
            val += pbar_cnt
            # progress_callback(val, pbar_cnt)
            progress_callback(pbar_cnt)
    else:
        val += 95
        # progress_callback(val, 95)
        progress_callback(95)

    # flush
    ds_xline.Destroy()
    driver = None
    
    # task-5
    # remove files
    directory1, filename1 = os.path.split(onshore_line)
    unlink_shapefiles(directory1, filename1.replace(".shp", "")) if file_format1 == "GeoJSON" else None
    directory2, filename2 = os.path.split(offshore_line)
    unlink_shapefiles(directory2, filename2.replace(".shp", "")) if file_format2 == "GeoJSON" else None
    # update progressbar
    val_rem = 100 - val
    # progress_callback(100, val_rem)
    progress_callback(val_rem)
    
    return
