# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 2016

@author: Debabrata Ghorai, Ph.D.

Regular image position shifting.

shift_side: N/S/E/W/NE/SE/SW/NW direction
"""

import numpy

from osgeo import gdal, gdalconst, gdal_array


def regular_shift_raster(in_raster, out_raster, shift_side=None, shift_dist=None):
    # raster data propertics
    driver = gdal.GetDriverByName('HFA')
    dataset = gdal.Open(in_raster, gdalconst.GA_ReadOnly)
    geot = dataset.GetGeoTransform()
    geop = dataset.GetProjection()

    # get output geo-transform
    if shift_side == 'N':
        outgeot = (geot[0], geot[1], geot[2], geot[3] +
                   shift_dist, geot[4], geot[5])
    elif shift_side == 'S':
        outgeot = (geot[0], geot[1], geot[2], geot[3] -
                   shift_dist, geot[4], geot[5])
    elif shift_side == 'E':
        outgeot = (geot[0]+shift_dist, geot[1],
                   geot[2], geot[3], geot[4], geot[5])
    elif shift_side == 'W':
        outgeot = (geot[0]-shift_dist, geot[1],
                   geot[2], geot[3], geot[4], geot[5])
    elif shift_side == 'NE':
        outgeot = (geot[0]+shift_dist, geot[1], geot[2],
                   geot[3]+shift_dist, geot[4], geot[5])
    elif shift_side == 'SE':
        outgeot = (geot[0]+shift_dist, geot[1], geot[2],
                   geot[3]-shift_dist, geot[4], geot[5])
    elif shift_side == 'SW':
        outgeot = (geot[0]-shift_dist, geot[1], geot[2],
                   geot[3]-shift_dist, geot[4], geot[5])
    elif shift_side == 'NW':
        outgeot = (geot[0]-shift_dist, geot[1], geot[2],
                   geot[3]+shift_dist, geot[4], geot[5])
    else:
        pass

    # write output
    array = numpy.array(dataset.GetRasterBand(1).ReadAsArray())
    result = gdal_array.OpenArray(array)
    result.SetGeoTransform(outgeot)
    result.SetProjection(geop)
    driver.CreateCopy(out_raster, result)
    return
