# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 19:29:41 2024

@author: Debabrata Ghorai, Ph.D.

Create buffer of a vector file.

"""

import geopandas as gpd
import pandas as pd
import shapely.ops as sp_ops

from pyproj import Transformer
from src.utils.ref_scripts import find_wgs2utm_epsg_code


def buffer_feature(input_feature, buffer_offset=None, src_epsg_code=None, save=False, out_buffer_feature=None):
    if isinstance(input_feature, str):
        gdf = gpd.read_file(input_feature)
    else:
        gdf = input_feature
    
    # check src projection
    if src_epsg_code == 4326:
        minx, miny, maxx, maxy = gdf.geometry.total_bounds
        dst_epsg_code = int(find_wgs2utm_epsg_code(minx, maxy))
        transformer1 = Transformer.from_crs(f'EPSG:{src_epsg_code}', f'EPSG:{dst_epsg_code}', always_xy=True)
        # set geometry - https://github.com/geopandas/geopandas/issues/1175
        geom_transformed1 = pd.Series([sp_ops.transform(transformer1.transform, geom) for geom in gdf['geometry'].tolist()])
        gdf.set_geometry(geom_transformed1, inplace=True, crs=dst_epsg_code)
        # create buffer
        gdf_buffer_series = gdf.buffer(buffer_offset)
        gdf_buffer = gpd.GeoDataFrame(geometry=gpd.GeoSeries(gdf_buffer_series))
        # re-project to src coordinates
        transformer2 = Transformer.from_crs(f'EPSG:{dst_epsg_code}', f'EPSG:{src_epsg_code}', always_xy=True)
        # set geometry
        geom_transformed2 = pd.Series([sp_ops.transform(transformer2.transform, geom) for geom in gdf_buffer['geometry'].tolist()])
        gdf_buffer.set_geometry(geom_transformed2, inplace=True, crs=src_epsg_code)
    else:
        gdf_buffer = gdf.buffer(buffer_offset)
    
    res = gdf_buffer.to_file(out_buffer_feature) if save else gdf_buffer
    
    return res
