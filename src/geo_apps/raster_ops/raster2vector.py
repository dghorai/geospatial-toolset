# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 2016

@author: Debabrata Ghorai, Ph.D.

Convert raster to point shapefile.
"""

from osgeo import gdal, gdalconst, gdalnumeric, ogr, osr


def raster_to_point(in_raster_data, out_point_file):
    # read raster image
    ds = gdal.Open(in_raster_data, gdalconst.GA_ReadOnly)
    # GetGeoTransform
    geot = ds.GetGeoTransform()
    # geop = ds.GetProjection()
    min_x, max_y = geot[0], geot[3]
    # get spatial reference from raster image
    sr = osr.SpatialReference()
    sr.ImportFromWkt(ds.GetProjection())
    # dataset to array
    array = gdalnumeric.LoadFile(in_raster_data)
    # check array dimension
    dim = array.ndim
    # create shapefile
    driver = ogr.GetDriverByName('ESRI Shapefile')
    shapeData = driver.CreateDataSource(out_point_file)
    # create layer
    lyr = shapeData.CreateLayer('point_layer', sr, ogr.wkbPoint)
    # add field
    if dim == 3:
        field_names = ["Band_"+str(i) for i in range(array.shape[0])]
        for field_name in field_names:
            lyr.CreateField(ogr.FieldDefn(field_name, ogr.OFTReal))
    else:
        lyr.CreateField(ogr.FieldDefn("DN", ogr.OFTReal))
    # lyr defn
    lyrDef = lyr.GetLayerDefn()
    # check array dimension
    if dim == 3:
        # raise CustomException("found multi-channel raster data")
        for i in range(array.shape[1]):
            for j in range(array.shape[2]):
                x = j*geot[1] + min_x + geot[1]/2.0
                y = i*geot[5] + max_y + geot[5]/2.0
                # create points
                point = ogr.Geometry(ogr.wkbPoint)
                point.SetPoint(0, x, y)
                # put point as a geometry inside a feature
                featIndex = 0
                feat = ogr.Feature(lyrDef)
                arr_values = list(array[:, i, j])
                for field_name, val in zip(field_names, arr_values):
                    feat.SetField(field_name, val)
                feat.SetGeometry(point)
                feat.SetFID(featIndex)
                # put feature in a layer
                lyr.CreateFeature(feat)
    else:
        # iterate over the numpy points
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                x = j*geot[1] + min_x + geot[1]/2.0
                y = i*geot[5] + max_y + geot[5]/2.0
                # create points
                point = ogr.Geometry(ogr.wkbPoint)
                point.SetPoint(0, x, y)
                # put point as a geometry inside a feature
                featIndex = 0
                feat = ogr.Feature(lyrDef)
                feat.SetField("DN", array[i][j])
                feat.SetGeometry(point)
                feat.SetFID(featIndex)
                # put feature in a layer
                lyr.CreateFeature(feat)
    # Flush
    shapeData.Destroy()
    return


def raster_to_line():
    pass


def raster_to_polygon():
    pass
