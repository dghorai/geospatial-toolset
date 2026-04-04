# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 10:08:54 2019

@author: Debabrata Ghorai, Ph.D.

Objective: Mosaic large GeoTiff files using GDAL and Python extension package with less time
"""

# Import Modules
import os
import time
import numpy
import ctypes
import subprocess

from osgeo import gdal
from osgeo.gdalconst import *
from logger import logger
from app import PRJ_DIR
from src.utils.geo_utils import timeit, write_geotiff_file


class LargeGeoTiffMosaic:
    """
    Mosaic large geotiff tiles using GDAL library and C/C++ Python extension.
    The C/C++ Python extention package written by Debabrata Ghorai to convert 
    float-type numpy array to integer-type numpy array fastly.
    """

    def __init__(self):
        """
        constructor
        """
        self.nodata = -9999
        # C/C++ Program for Python Extension
        self.mylib = ctypes.CDLL(os.path.join(PRJ_DIR, 'artifacts', 'extension', 'library_windows64bit', 'floattoint.dll'))
        self.Multiply_and_Convert_Values = self.mylib.Multiply_and_Convert_Values
        self.Multiply_and_Convert_Values.restype = None
        self.Multiply_and_Convert_Values.argtypes = [ctypes.c_int, ctypes.c_int, numpy.ctypeslib.ndpointer(
            ctypes.c_float), ctypes.c_float, numpy.ctypeslib.ndpointer(ctypes.c_int)]

    def floatraster_to_intraster(self, tif_file_list, outf):
        """
        Convert float-type raster to integer-type raster.
        """
        cnt = 1
        new_file_list = list()
        for f in tif_file_list:
            print("\t"+str(f))
            ds = gdal.Open(f, GA_ReadOnly)
            arr = ds.GetRasterBand(1).ReadAsArray()
            out_int = os.path.splitext(outf)[0]+str(cnt)+'_int.tif'
            out_arr = numpy.zeros(arr.shape, dtype=numpy.int32)
            self.Multiply_and_Convert_Values(
                arr.shape[0], arr.shape[1], arr, 1000, out_arr)
            out_arr[numpy.logical_or(
                out_arr <= -2147483648, out_arr > 2147483647)] = self.nodata
            # write output
            gt = ds.GetGeoTransform()
            sr = ds.GetProjection()
            write_geotiff_file(out_arr, gt=gt, sr=sr,
                               outfile_path=out_int, dtype=gdal.GDT_Int32)
            new_file_list.append(out_int)
            del out_arr
            del arr
            ds = None
            cnt += 1
        return new_file_list

    def get_geotiff_files(self, path):
        """
        Get all the geotiff files into a list.
        """
        os.chdir(path)
        os.getcwd()
        tif_file_list = list()
        for root, dirs, files in os.walk("."):
            for d in dirs:
                _ = os.path.relpath(os.path.join(root, d), ".")
            for f in files:
                anyfile = os.path.join(
                    path, os.path.relpath(os.path.join(root, f), "."))
                # filter only tif files
                if os.path.splitext(anyfile)[-1] == '.tif':
                    tif_file_list.append(anyfile)
        return tif_file_list

    def mosaic_raster_tiles(self, path, outf):
        """
        Mosaic all the geotiff files.
        """
        t1 = time.time()
        logger.info("Get GeoTiff Files")
        tif_file_list = self.get_geotiff_files(path)
        timeit("\tTime: ", time.time() - t1)

        t2 = time.time()
        logger.info("Multiply Values by 1000 and Convert to Int")
        new_file_list = self.floatraster_to_intraster(tif_file_list, outf)
        timeit("\tTime: ", time.time() - t2)

        t3 = time.time()
        logger.info("Combine into a Single File")
        vrt_output = os.path.splitext(outf)[0]+'.vrt'
        gdal.BuildVRT(vrt_output, new_file_list)
        timeit("\tTime: ", time.time() - t3)

        t4 = time.time()
        logger.info(
            "Export GeoTiff with Original Reference System and LZW Compression")
        tif_output = os.path.splitext(outf)[0]+'.tif'
        subprocess.call(
            "gdal_translate -of GTiff -co COMPRESS=LZW "+vrt_output+" "+tif_output)
        timeit("\tTime: ", time.time() - t4)

        t5 = time.time()
        logger.info("Pyramids Built")
        image = gdal.Open(tif_output, 0)
        gdal.SetConfigOption('COMPRESS_OVERVIEW', 'DEFLATE')
        image.BuildOverviews("NEAREST", [2, 4, 8, 16, 32, 64])
        del image
        timeit("\tTime: ", time.time() - t5)

        # Delete intermediate files
        for d in new_file_list:
            os.remove(d)
        # remove vrt file
        if os.path.isfile(vrt_output):
            os.remove(vrt_output)
        else:
            logger.info("Error: {} file not found".format(vrt_output))
        return


def mosaic_raster(raster_path, out_mosaic):
    obj = LargeGeoTiffMosaic()
    obj.mosaic_raster_tiles(raster_path, out_mosaic)
    return
