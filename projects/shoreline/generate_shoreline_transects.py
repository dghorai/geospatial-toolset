# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 21:29:13 2024

@author: Debabrata Ghorai, Ph.D.

Generate shoreline transects.

"""

from osgeo import ogr
from src.utils.ref_scripts import fixed_interval_points
from src.geo_apps.vector_ops.find_nearest_point import get_nearest_point


def create_shoreline_transects(onshore_line=None, offshore_line=None, out_transect_line=None, x_interval=None):
    driver = ogr.GetDriverByName("ESRI Shapefile")
    ds_base = driver.Open(onshore_line, 0)
    lyr = ds_base.GetLayer(0)
    sr = lyr.GetSpatialRef()
    sr.ExportToWkt()
    ds_xline = driver.CreateDataSource(out_transect_line)
    lyr = ds_xline.CreateLayer('myLyr', sr, ogr.wkbMultiLineString)
    lyrDef = lyr.GetLayerDefn()

    baseline_points = fixed_interval_points(
        onshore_line, x_interval, flipline=True)

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
    # flush
    ds_xline.Destroy()
    driver = None
    return


# if __name__ == '__main__':
#     point_interval = 500
#     shore_baseline = ''  # baseline
#     shore_referenceline = ''  # reference line
#     out_crosssection_line = ""  # Output Transect File
#     create_shoreline_transects(
#         onshore_line=shore_baseline,
#         offshore_line=shore_referenceline,
#         out_transect_line=out_crosssection_line,
#         x_interval=point_interval
#     )
