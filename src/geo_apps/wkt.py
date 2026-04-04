# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 2022

@author: Debabrata Ghorai, Ph.D.

WKT related operation.
"""

import warnings
import pandas as pd
import geopandas as gpd
import numpy as np

from logger import CustomException
from src.utils.ref_scripts import (
    line_intersection,
    ray_tracing_method,
    sort_counter_clockwise
)


warnings.filterwarnings(action='ignore')


class WktUtils:
    """
    A class for common and useful GIS tools.
    This class consisting of several GIS functions and this functions using as a helper function
    for various geospatial application development.
    """

    def __init__(self):
        """constructor"""
        # 1 m2 = 0.000247105 acre
        self.sqm2acre = 0.000247105

    def wkt_to_gdf(self, wkts, crs='epsg:4326'):
        """Create WKT string to GeoDataFrame"""
        # geom_str to geodataframe
        df = pd.DataFrame([wkts], columns=['geometry'])
        df['geometry'] = gpd.GeoSeries.from_wkt(df['geometry'])
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=crs)
        return gdf

    def wkt_vertices(self, wkts):
        """Extract nodes of a WKT"""
        coords_list_fn = None
        if type(wkts) == str:
            print('Input is wkt string')
            if wkts.split()[0] == 'POLYGON':
                # extract all coordinates from wkt
                coords_str = wkts.strip().replace('POLYGON ((', '').replace('))', '').split(',')
                coords_list = [[float(i) for i in ii.split()]
                               for ii in coords_str]
                if coords_list[0] == coords_list[-1]:
                    coords_list_fn = coords_list
                else:
                    coords_list_fn = []
                    raise Exception("Invalid WKT or polygon not closed.")
            else:
                coords_list_fn = []
                raise CustomException(
                    "Only POLYGON types are allowed or WKT string may be incorrect.")
        else:
            print('Input is geodataframe')
            # Extract nodes from GeoDataFrame
            g = [i for i in wkts.geometry]
            x, y = g[0].exterior.coords.xy
            coords_list_fn = np.dstack((x, y)).tolist()
        return coords_list_fn

    def extract_overlap_polygon(self, poly1, poly2):
        """Extract overlap polygon vertices"""
        poly1_geom = self.wkt_vertices(poly1)
        poly2_geom = self.wkt_vertices(poly2)
        # extract overlap polygon
        overlap_nodes = []
        # find polygon-1 nodes inside polygon-2
        for point in poly1_geom:
            output = ray_tracing_method(point, poly2_geom)
            if output == True:
                overlap_nodes.append(point)
        # find polygon-2 nodes inside polygon-1
        for point in poly2_geom:
            output = ray_tracing_method(point, poly1_geom)
            if output == True:
                overlap_nodes.append(point)
        # find intersection nodes
        line_segment_poly1 = [[poly1_geom[j], poly1_geom[j+1]]
                              for j in range(len(poly1_geom)-1)]
        line_segment_poly2 = [[poly2_geom[j], poly2_geom[j+1]]
                              for j in range(len(poly2_geom)-1)]
        # loop over list of nodes
        for line1 in line_segment_poly1:
            for line2 in line_segment_poly2:
                point = line_intersection(line1, line2)
                if point != None:
                    overlap_nodes.append(point)
        # create final polygon node list from list of nodes
        overlap_nodes_unique = list(map(list, set(map(tuple, overlap_nodes))))
        # sort list of nodes
        nodes = sort_counter_clockwise(overlap_nodes_unique)
        # append first node to create close polygon
        poly_overlap_nodes = nodes + [nodes[0]]
        return poly_overlap_nodes
