# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 21:29:13 2024

@author: Debabrata Ghorai, Ph.D.

Generate sequence ID of shoreline transects.
This project only handle shapefile.

"""

import os
import geopandas as gpd

from pathlib import Path
from osgeo import ogr
from src.geo_apps.vector_ops.lines_middle_points import generate_lines_middle_point
from src.geo_apps.vector_ops.create_buffer import buffer_feature
from src.utils.geo_utils import (
    select_by_location,
    reading_polyline,
    get_shapefile_epsg_code,
    split_lines_at_points,
    unlink_files
)


def create_line_seqid(line_feature, out_line_feature):
    # Input shapefile layer
    shapefile = ogr.Open(line_feature)
    layer = shapefile.GetLayer(0)
    # Get input file spatial reference
    sr = layer.GetSpatialRef()
    sr.ExportToWkt()
    # Create new shapefile
    driver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(out_line_feature):
        driver.DeleteDataSource(out_line_feature)
    # create data source
    ds = driver.CreateDataSource(out_line_feature)
    # Create new shapefile layer
    lyr = ds.CreateLayer('mylyr1', sr, ogr.wkbPoint)
    lyrDef = lyr.GetLayerDefn()
    # Add the Sequence ID Field
    lyr.CreateField(ogr.FieldDefn("SeqID", ogr.OFTInteger))

    # Loop over the input features
    for i in range(layer.GetFeatureCount()):
        feature = layer.GetFeature(i)
        geom = feature.GetGeometryRef()
        pnts = geom.GetPoints()
        sgmnt_end = pnts[-1]
        # Point file generation
        point = ogr.Geometry(ogr.wkbPoint)
        point.SetPoint(0, sgmnt_end[0], sgmnt_end[1])
        featIndex = 0
        feat = ogr.Feature(lyrDef)
        feat.SetGeometry(point)
        feat.SetFID(featIndex)
        feat.SetField("SeqID", i+1)
        lyr.CreateFeature(feat)
    # Flush
    ds.Destroy()
    ds = None
    return


def yield_transects_seqid(
        temp_dir, baseline, transects_lines, 
        out_transects_with_seqid, 
        col_name=None
):
    """Generate sequence ID of shoreline transects"""
    # define temporary files
    out_transects = Path(transects_lines)
    out_union_line = Path(os.path.join(temp_dir, 'tests', 'results', 'union_xline.shp'))
    out_mid_points = Path(os.path.join(temp_dir, 'tests', 'results', 'union_xline_midpoints.shp'))
    out_union_line_selected = Path(os.path.join(temp_dir, 'tests', 'results', 'union_xline_selected.shp'))
    out_seqid_xpoints = Path(os.path.join(temp_dir, 'tests', 'results', 'xpoints_seqid.shp'))

    # check file exists or not
    out_transects.parent.mkdir(parents=True, exist_ok=True)
    out_union_line.parent.mkdir(parents=True, exist_ok=True)
    out_mid_points.parent.mkdir(parents=True, exist_ok=True)
    out_union_line_selected.parent.mkdir(parents=True, exist_ok=True)
    out_seqid_xpoints.parent.mkdir(parents=True, exist_ok=True)

    # convert to realpath
    out_transects = os.path.realpath(out_transects)
    out_union_line = os.path.realpath(out_union_line)
    out_mid_points = os.path.realpath(out_mid_points)
    out_union_line_selected = os.path.realpath(out_union_line_selected)
    out_seqid_xpoints = os.path.realpath(out_seqid_xpoints)

    # get src epsg code
    src_epsg_code = get_shapefile_epsg_code(out_transects)

    # merge and split shoreline and trsancets lines
    # https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.html
    _, geoms = reading_polyline(out_transects)

    points = []
    for geom in geoms:
        points.append(geom[0])
        points.append(geom[-1])

    gdf_segments = split_lines_at_points(
        baseline, points, 
        snap_dist=10, 
        src_epsg_code=src_epsg_code
    )
    gdf2 = gpd.read_file(out_transects)

    # union line layers
    res_union = gpd.overlay(
        gdf_segments, gdf2, 
        how='union', 
        keep_geom_type=True
    )
    res_union.to_file(out_union_line)

    # create mid-point of lines
    generate_lines_middle_point(
        line_feature=out_union_line, 
        out_mid_points=out_mid_points
    )

    # Create 1 meter buffer for Baseline
    baseline_buffer = buffer_feature(
        baseline, 
        buffer_offset=1.0, 
        src_epsg_code=src_epsg_code
    )
    baseline_buffer_gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(baseline_buffer))
    # Select points based on the Buffer Baseline Polygon
    point_in_poly = select_by_location(
        select_feature_from=out_mid_points, 
        feature_overlap_with=baseline_buffer_gdf, 
        overlap_method='within'
    )
    # Create 1 meter buffer for selected point file
    spoints_buffer = buffer_feature(
        point_in_poly, 
        buffer_offset=1.0, 
        src_epsg_code=src_epsg_code
    )
    spoints_buffer_gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(spoints_buffer))

    # Select Base-Trans-Splited files based on selected points file polygon
    select_by_location(
        select_feature_from=out_union_line,
        feature_overlap_with=spoints_buffer_gdf,
        overlap_method='intersects',
        save=True,
        out_selected_features=out_union_line_selected
    )

    # assign sequence id to shoreline points
    create_line_seqid(out_union_line_selected, out_seqid_xpoints)

    # k) Create 1 meter buffer on the (j) created point file
    seqid_point_buffer = buffer_feature(
        out_seqid_xpoints, 
        buffer_offset=1.0, 
        src_epsg_code=src_epsg_code
    )
    seqid_point_buffer_gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(seqid_point_buffer))

    # l) Spatial join between transects and (k) feature class
    select_by_location(
        select_feature_from=out_transects,
        feature_overlap_with=seqid_point_buffer_gdf,
        overlap_method='intersects',
        save=True,
        out_selected_features=out_transects_with_seqid,
        col_name=col_name
    )
    
    # remove temporary files
    unlink_files(is_file=True, file_path=out_union_line)
    unlink_files(is_file=True, file_path=out_mid_points)
    unlink_files(is_file=True, file_path=out_union_line_selected)
    unlink_files(is_file=True, file_path=out_seqid_xpoints)
    return
