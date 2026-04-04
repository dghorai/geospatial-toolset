# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 2016

@author: Debabrata Ghorai, Ph.D.

Digital Number to Radiance Conversion.
"""

import numpy
from osgeo import gdal, gdal_array


def dn_to_radiance(input_image, output_folder, lmax_list, lmin_list, qcal_min, qcal_max, prefix="band"):
    """
    Digital Number to Radiance Conversion.
    """
    # set output file projection from input file
    ds = gdal.Open(input_image)
    geotransform = ds.GetGeoTransform()
    prj = ds.GetProjection()
    # get total input raster bands (count)
    inrasband = [band+1 for band in range(ds.RasterCount)]
    # set output file driver
    # HFA for *.img reading and writting
    outdriver = gdal.GetDriverByName('GTiff')
    # processing
    for i in inrasband:
        dnfile = ds.GetRasterBand(i)
        dn = numpy.array(dnfile.ReadAsArray())
        radiance = ((lmax_list[i-1] - lmin_list[i-1]) /
                    (qcal_max - qcal_min))*dn + lmin_list[i-1]
        # create output file with projection
        radresult = gdal_array.OpenArray(radiance)
        radresult.SetGeoTransform(geotransform)
        radresult.SetProjection(prj)
        # print ("Digital number to radiance conversion for band : %d" % i)
        outdriver.CreateCopy(output_folder+"\\"+prefix +
                             "_"+str(i)+".img", radresult)
    outdriver = None
    ds = None
    return
