# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 2016

@author: Debabrata Ghorai, Ph.D.

Radiance to degree celsious conversion.

GDAL raster drivers: https://gdal.org/drivers/raster/index.html
"""

import numpy
from osgeo import gdal, gdal_array


def radiance2degree_celsious_temperature(thermal_band, output_folder, prefix="atmradi_band", gdal_rst_driver='GTiff'):
    # set output file projection from input file
    ds = gdal.Open(thermal_band)
    gt = ds.GetGeoTransform()
    sr = ds.GetProjection()

    # get total input raster bands (count)
    bands = [band+1 for band in range(ds.RasterCount)]

    # degree Celsiuos conversion parameters
    k2 = 1260.56  # TODO: degrees Kelvin constant for LT5-TM
    k1 = 607.76  # TODO: degrees Kelvin constant for LT5-TM

    # set output file driver
    # HFA for *.img reading and writting
    out_driver = gdal.GetDriverByName(gdal_rst_driver)

    # processing
    for b in bands:
        atmradfile = ds.GetRasterBand(b)
        atmradi = numpy.array(atmradfile.ReadAsArray())
        var1 = (k1/atmradi)
        var2 = numpy.log(var1+1)
        degreeKelvin = (k2/var2)
        degreeCelsious = degreeKelvin - 273.0
        # create output file with projection
        dcresult = gdal_array.OpenArray(degreeCelsious)
        dcresult.SetGeoTransform(gt)
        dcresult.SetProjection(sr)
        # print("Temperature in degree Celsious for band : %d" % i)
        out_driver.CreateCopy(output_folder+"\\"+prefix +
                              "_"+str(b)+".img", dcresult)
    return
