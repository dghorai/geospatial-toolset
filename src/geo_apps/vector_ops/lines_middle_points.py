# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 21:29:13 2024

@author: Debabrata Ghorai, Ph.D.

Generate middle point of a line.

"""

import os
import glob

from osgeo import ogr
from src.utils.ref_scripts import interval_point
from src.utils.geo_utils import dist_calc


def generate_lines_middle_point(line_feature=None, out_mid_points=None):
    src_ds = ogr.Open(line_feature)
    layer = src_ds.GetLayer(0)
    # Get input file spatial reference
    sr = layer.GetSpatialRef()
    # sr.ExportToWkt()
    # Create new shapefile
    driver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(out_mid_points):
        try:
            driver.DeleteDataSource(out_mid_points)
        except:
            pass
        flist = glob.glob(
            os.path.join(
                os.path.dirname(out_mid_points),
                os.path.basename(out_mid_points).split('.')[0]+'*'
            )
        )
        for f in flist:
            os.remove(f)
    # create ds
    ds = driver.CreateDataSource(out_mid_points)
    # Create new shapefile layer
    lyr = ds.CreateLayer('mylyr', sr, ogr.wkbPoint)
    lyrDef = lyr.GetLayerDefn()

    # calculate mid point of a line
    for i in range(layer.GetFeatureCount()):
        feature = layer.GetFeature(i)
        geom = feature.GetGeometryRef()
        pnts = geom.GetPoints()
        if not isinstance(pnts, type(None)):
            # calc segment dist
            lineDist = list()
            for p in range(len(pnts)-1):
                lineDist.append(dist_calc(pnts[p], pnts[p+1]))
            # get mid-length
            midLength = sum(lineDist)*0.5
            objects = [i+1]
            nodes = [pnts]
            points, _ = interval_point(objects, nodes, midLength)
            # Point file generation
            point = ogr.Geometry(ogr.wkbPoint)
            point.SetPoint(0, points[0][0], points[0][1])
            featIndex = 0
            feat = ogr.Feature(lyrDef)
            feat.SetGeometry(point)
            feat.SetFID(featIndex)
            lyr.CreateFeature(feat)
    # Flush
    layer = None
    sr = None
    ds.Destroy()
    ds = None
    driver = None
    lyrDef = None
    lyr = None
    return
