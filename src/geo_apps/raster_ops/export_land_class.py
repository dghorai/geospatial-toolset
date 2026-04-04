# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 2022

@author: Debabrata Ghorai, Ph.D.

Export specific LULC class as raster file.
"""

import numpy
from osgeo import gdal, gdalconst, gdalnumeric, gdal_array


def extract_lulc_class(lulc_raster, out_raster, class_number=0):
    """Export specific lulc class as raster file"""
    # raster data propertics
    driver = gdal.GetDriverByName('HFA')  # for *.img file
    dataset = gdal.Open(lulc_raster, gdalconst.GA_ReadOnly)
    geot = dataset.GetGeoTransform()
    geop = dataset.GetProjection()
    # convert raster to numpy array
    try:
        array = gdalnumeric.LoadFile(lulc_raster)
    except:
        array = numpy.array(dataset.GetRasterBand(1).ReadAsArray())
    # iterate over the array
    newArray = numpy.zeros(array.shape)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i, j] == class_number:
                newArray[i, j] = class_number
            del j
        del i
    # write new raster
    result = gdal_array.OpenArray(newArray)
    result.SetGeoTransform(geot)
    result.SetProjection(geop)
    driver.CreateCopy(out_raster, result)
    return
