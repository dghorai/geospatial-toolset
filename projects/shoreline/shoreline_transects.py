# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 21:29:13 2024

@author: Debabrata Ghorai, Ph.D.

Generate shoreline transects.
This project only handle shapefile and geojson file.

"""

import os

from osgeo import ogr
from src.utils.ref_scripts import fixed_interval_points
from src.geo_apps.vector_ops.find_nearest_point import get_nearest_point
from src.utils.geo_utils import (
    convert_to_shapefile, 
    check_vector_file_format,
    unlink_shapefiles
)


def create_shoreline_transects(
        onshore_line=None, 
        offshore_line=None, 
        out_transect_line=None, 
        x_interval=None, 
        progress_callback=None
):
    # initilize pbar value
    val = 0

    # check file format
    file_format1 = check_vector_file_format(onshore_line)
    onshore_line = convert_to_shapefile(
        src_file=onshore_line, 
        dest_file=onshore_line.replace(".geojson", ".shp")
    ) if file_format1 == "GeoJSON" else onshore_line
    
    file_format2 = check_vector_file_format(offshore_line)
    offshore_line = convert_to_shapefile(
        src_file=offshore_line, 
        dest_file=offshore_line.replace(".geojson", ".shp")
    ) if file_format2 == "GeoJSON" else offshore_line
    # update progressbar
    progress_callback(1)
    val += 1
    
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
    progress_callback(1)
    val += 1
    
    # generate interval points
    baseline_points = fixed_interval_points(
        onshore_line, 
        x_interval, 
        flipline=True
    )
    # update progressbar
    progress_callback(1)
    val += 1
    
    # check points and create transects
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
            progress_callback(pbar_cnt)
            val += pbar_cnt            
    else:
        progress_callback(95)
        val += 95

    # flush
    ds_xline.Destroy()
    driver = None
    
    # remove files
    directory1, filename1 = os.path.split(onshore_line)
    unlink_shapefiles(
        directory1, 
        filename1.replace(".shp", "")
    ) if file_format1 == "GeoJSON" else None
    directory2, filename2 = os.path.split(offshore_line)
    unlink_shapefiles(
        directory2, 
        filename2.replace(".shp", "")
    ) if file_format2 == "GeoJSON" else None
    # update progressbar
    val_rem = 100 - val
    progress_callback(val_rem)
    
    return
