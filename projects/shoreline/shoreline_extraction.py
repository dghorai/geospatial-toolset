#!/usr/bin/env python
"""
****************************************************************************
* File Name:      shoreline_extraction.py
* Created:        January 1, 2019
* Author:         Debabrata Ghorai, Ph.D.
* Purpose:        Automatic Shoreline Extraction
* Description:    Shoreline Analysis, Change Detection, etc.
* Research Paper: Ghorai, D., & Mahapatra, M. (2020). Extracting Shoreline from Satellite Imagery for GIS Analysis.
                  Remote Sensing in Earth Systems Sciences, 1-10. DOI 10.1007/s41976-019-00030-w.
* Revesions:      June 3, 2020: Header information added
                  March 22, 2024: Added this application into package
****************************************************************************
"""

import os
import time
import numpy
import scipy.ndimage
import scipy

from osgeo import gdal
from osgeo.gdalconst import *
from logger import logger
from src.utils.geo_utils import (
    timeit,
    OTSU,
    write_geotiff_file
)


class SeaWaterClass:
    """Sea water and land classification"""

    def __init__(self, output_folder, blue_band_path, swir2_band_path, gray_level):
        self.output_folder = output_folder
        self.blue_band_path = blue_band_path
        self.swir2_band_path = swir2_band_path
        self.gray_level = gray_level

    def Mean_Filter_3x3(self, arr):
        return scipy.ndimage.uniform_filter(arr, size=3)

    def Get_MinVal_NextToZero(self, arr):
        nMin = None
        minv = numpy.nanmin(arr)
        if minv == 0:
            arrCopy = numpy.copy(arr)
            arrCopy[arrCopy == 0] = numpy.nan
            nMin = numpy.nanmin(arrCopy)
            del arrCopy
        else:
            nMin = minv
        return nMin

    def Coastal_Water_Index(self, arr_blue, arr_swir2):
        mval = self.Get_MinVal_NextToZero(arr_swir2)
        arr_swir2[arr_swir2 == 0] = mval
        max_b2, max_b7 = numpy.nanmax(arr_blue), numpy.nanmax(arr_swir2)
        return (max_b7*arr_blue)/(max_b2*arr_swir2)

    def Bright_Object_Enhancement(self, arr):
        # Custom filter created by Debabrata Ghorai (Convolution Matrix Filter)
        F = numpy.array([[2, 2, 2], [2, 16, 2], [2, 2, 2]])
        # Add rows
        AR = numpy.vstack(
            (numpy.array(arr[0, :]), arr, numpy.array(arr[-1, :])))
        # Add columns
        arr_2 = numpy.hstack((numpy.array(AR[:, 0]).reshape(
            AR.shape[0], 1), AR, numpy.array(AR[:, -1]).reshape(AR.shape[0], 1)))
        # Get copy
        arr_3 = numpy.copy(arr_2)  # use for loop
        # Loop over array
        for i in range(arr_3.shape[0]):
            for j in range(arr_3.shape[1]):
                K = arr_3[i-1:i+2, j-1:j+2]
                if K.shape == (3, 3):
                    if arr_3[i][j] == 0:
                        pass
                    else:
                        arr_2[i][j] = (F*K).sum()
                del j
            del i
        del arr_3
        # Get new array
        NewArr = arr_2[1:-1, 1:-1]
        # Return final array
        return NewArr

    def Classification_OTSU(self, arr, graylevel):
        # (1) min and max
        mx, mn = numpy.nanmax(arr), numpy.nanmin(arr)
        # (2) rescale
        arr[numpy.isinf(arr)] = mn
        arr[numpy.isnan(arr)] = mn
        arr[arr < mn] = mn
        arr_2 = (arr - mn)/(mx - mn)
        del arr
        arr_3 = arr_2*graylevel
        del arr_2
        arr_4 = arr_3.astype(numpy.int)
        del arr_3
        # (3) otsu
        t = OTSU(arr_4, graylevel)
        # (4) apply otsu
        arr_4[arr_4 <= t] = 0
        arr_4[arr_4 > t] = 1
        del t
        return arr_4

    def Image_Morphological_Filter(self, arrCwi4):
        # (5) morphology opening
        arrCwi5 = scipy.ndimage.binary_opening(
            arrCwi4, structure=numpy.ones((3, 3))).astype(numpy.int)
        del arrCwi4
        # (6) morphology closing
        arrCwi6 = scipy.ndimage.binary_closing(
            arrCwi5, structure=numpy.ones((3, 3))).astype(numpy.int)
        del arrCwi5
        return arrCwi6

    def Get_MaxCount_PixVal(self, nBlobs, aBlobs):
        # Get the pixel value which is highest in counting
        pMax = 0
        pVal = None
        # aBlobs = array of blobs
        # nBlobs = number of blobs
        for i in range(1, nBlobs+1):
            nPix = len(aBlobs[aBlobs == i])
            if nPix > pMax:
                pMax = nPix
                pVal = i
            del i
        return pVal, pMax

    def Get_SeaWater_LandMass(self, arr, graylevel):
        # print ("\tCounting the number of groups of 1s in a boolean map of array")
        blobs, number_of_blobs = scipy.ndimage.label(arr)
        del arr
        # print ("\tGet the pixel value which is highest in counting")
        pval, pmax = self.Get_MaxCount_PixVal(number_of_blobs, blobs)
        # print ("\tKeep pVal and replace all other values to 0")
        arrCwi2 = numpy.where(blobs == pval, blobs, 0)
        # plt.imshow(blobs, cmap='gray', interpolation='nearest')
        del blobs, number_of_blobs, pval, pmax
        return arrCwi2

    def Write_GeoTiff_Output(self, arr, dsOut, outfile):
        # in gdal actual rows == cols and actual cols == rows
        [cols, rows] = arr.shape
        trans = dsOut.GetGeoTransform()
        proj = dsOut.GetProjection()
        write_geotiff_file(arr, gt=trans, sr=proj,
                           outfile_path=outfile, dtype=gdal.GDT_Float32)
        return

    def Read_BlueAndSWIR2_Band(self, blue_band, swir2_band):
        # print ("Register all of the drivers")
        gdal.AllRegister()
        logger.info("Read image")
        dsBlue = gdal.Open(blue_band, GA_ReadOnly)
        dsSwir2 = gdal.Open(swir2_band, GA_ReadOnly)
        logger.info("Convert raster to numpy array")
        bArr = dsBlue.ReadAsArray()
        sArr = dsSwir2.ReadAsArray()
        logger.info("Check if any +/- inf presents")
        bArr[numpy.isinf(bArr)] = numpy.nan
        sArr[numpy.isinf(sArr)] = numpy.nan
        logger.info("Replace None by 0")
        bArr[numpy.isnan(bArr)] = 0
        sArr[numpy.isnan(sArr)] = 0
        logger.info("3x3 Mean Filter")
        bMeanf = self.Mean_Filter_3x3(bArr)
        sMeanf = self.Mean_Filter_3x3(sArr)
        return bMeanf, sMeanf

    def Get_All_Arrays(self, blue_band, swir2_band):
        t1 = time.time()
        bMeanf2, sMeanf2 = self.Read_BlueAndSWIR2_Band(blue_band, swir2_band)
        timeit("\tTime: ", time.time() - t1)
        t2 = time.time()

        logger.info("Coastal Water Index")
        cwi = self.Coastal_Water_Index(bMeanf2, sMeanf2)
        timeit("\tTime: ", time.time() - t2)
        t3 = time.time()

        logger.info("CWI Image Enhancement")
        eCwi = self.Bright_Object_Enhancement(cwi)
        timeit("\tTime: ", time.time() - t3)
        t4 = time.time()

        logger.info("Image Segmentation using OTSU Thresholding")
        otsuCls = self.Classification_OTSU(eCwi, self.gray_level)
        timeit("\tTime: ", time.time() - t4)
        t5 = time.time()

        logger.info("Image Morphological Filter")
        mfCls = self.Image_Morphological_Filter(otsuCls)
        timeit("\tTime: ", time.time() - t5)
        t6 = time.time()

        logger.info("Get Sea Water and Land Mass")
        arrSea = self.Get_SeaWater_LandMass(mfCls, self.gray_level)
        timeit("\tTime: ", time.time() - t6)
        return cwi, eCwi, otsuCls, mfCls, arrSea

    def GenerateRasters(self):
        # Output Files
        OutCWI = os.path.join(self.output_folder, "Result_CWI.tif")
        OutEnCWI = os.path.join(self.output_folder, "Result_EnhancedCWI.tif")
        OutOtsuCls = os.path.join(self.output_folder, "Result_OtsuClasses.tif")
        OutMorphoFlt = os.path.join(self.output_folder, "Result_MF.tif")
        OutSeaLand = os.path.join(self.output_folder, "Result_SeaLandMass.tif")

        t0 = time.time()
        CWI, eCWI, OtsuCls, MF, SeaLand = self.Get_All_Arrays(
            self.blue_band_path, self.swir2_band_path)
        logger.info("Write Outputs")
        ds = gdal.Open(self.blue_band_path, GA_ReadOnly)
        self.Write_GeoTiff_Output(CWI, ds, OutCWI)
        self.Write_GeoTiff_Output(eCWI, ds, OutEnCWI)
        self.Write_GeoTiff_Output(OtsuCls, ds, OutOtsuCls)
        self.Write_GeoTiff_Output(MF, ds, OutMorphoFlt)
        self.Write_GeoTiff_Output(SeaLand, ds, OutSeaLand)
        timeit("Total time: ", time.time() - t0)
        return


def generate_shoreline_raster(output_dir, blue_band_path, swir2_band_path, gray_levels=None):
    """
    This function will generate five raster files including sea-land mass raster.
    The sea-land mass raster is the final shoreline raster file and all others are 
    intermediate files for visualization and can be removed.
    """
    if isinstance(gray_levels, type(None)):
        gray_levels = 255
    sea = SeaWaterClass(output_dir, blue_band_path,
                        swir2_band_path, gray_levels)
    sea.GenerateRasters()
    return
