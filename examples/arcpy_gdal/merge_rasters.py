# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 18:53:49 2023

@author: Debabrata Ghorai, PhD.

Merge single band rasters.

"""

import os
import glob
import numpy as np

from osgeo import gdal



class MergeRasters:
    """Merge overlap/non-overlap single band rasters"""
    
    def __init__(self):
        pass
    
    def read_raster(self, rfile):
        ds = gdal.Open(rfile)
        gt = ds.GetGeoTransform()
        minx = gt[0]
        maxy = gt[3]
        maxx = minx + gt[1] * ds.RasterXSize
        miny = maxy + gt[5] * ds.RasterYSize
        return ds, gt, minx, maxy, maxx, miny
    
    def save_raster(self, arr, gt=None, sr=None, outfile=None):
        nr, nc = arr.shape
        # check files
        if os.path.exists(outfile):
            os.remove(outfile)
        # outdriver
        driver = gdal.GetDriverByName("GTiff")
        outdata = driver.Create(outfile, nc, nr, 1, gdal.GDT_UInt16)
        outdata.SetGeoTransform(gt)
        outdata.SetProjection(sr)
        outdata.GetRasterBand(1).WriteArray(arr)
        outdata.FlushCache()
        outdata = None
        return
    
    def calc_merge_raster_params(self, rlist):
        minxs, maxys, maxxs, minys = [], [], [], []
        for f in rlist:
            ds, gt, minx, maxy, maxx, miny = self.read_raster(f)
            minxs.append(minx)
            maxys.append(maxy)
            maxxs.append(maxx)
            minys.append(miny)
        # global extension
        min_x, max_y, max_x, min_y = min(minxs), max(maxys), max(maxxs), min(minys)
        # total pixel size
        rows = int(round((max_y - min_y)/(-gt[5]), 0))
        cols = int(round((max_x - min_x)/gt[1], 0))
        # geo-transform of new raster
        out_gt = list(gt)
        out_gt[0], out_gt[3] = min_x, max_y
        # get spatial reference of raster
        out_sr = ds.GetProjection()
        # make ds is none
        ds = None
        return min_x, max_y, rows, cols, out_gt, out_sr
    
    def create_raster_ndarray(self, rlist, band_type='single'):
        # get mosaic parameters
        min_x, max_y, rows, cols, out_gt, out_sr = self.calc_merge_raster_params(rlist)
        # create n-d mask array
        n = len(rlist)
        mask_array = np.zeros((n, rows, cols), dtype=int)
        if band_type == 'single':
            for i, f in enumerate(rlist):
                ds, gt, minx, maxy, maxx, miny = self.read_raster(f)
                arr = ds.GetRasterBand(1).ReadAsArray()
                # get array index
                rows1, cols1 = arr.shape
                min_x2, max_y2, max_x2, min_y2 = min_x, max_y, maxx, miny
                rows2 = int(round((max_y2 - min_y2)/(-gt[5]), 0))
                cols2 = int(round((max_x2 - min_x2)/gt[1], 0))
                i1, j1 = rows2 - rows1, cols2 - cols1
                i2, j2 = rows2, cols2
                # appened array to mask_array
                mask_array[i, i1:i2, j1:j2] = arr
        else:
            raise "Convert multi-band to individual band before running this function."
        return mask_array, out_gt, out_sr
    
    def merge_rasters(self, rpath, outfile=None, is_overlap=True, overlap_criteria=None):
        rlist = glob.glob(os.path.join(rpath, "*.tif"))
        if is_overlap:
            mask_array, out_gt, out_sr = self.create_raster_ndarray(rlist)
            # comvert multi-dimention array to 2d array with stats of every pixel
            if overlap_criteria == 'max':
                new_arr = np.max(mask_array, axis=0)
            else:
                raise "Add overlap_criteria here"
            # save raster
            self.save_raster(new_arr, gt=out_gt, sr=out_sr, outfile=outfile)
        else:
            ds = gdal.Warp(outfile, rlist, format="GTiff", options=["COMPRESS=LZW", "TILED=YES"])  # if you want
            ds.FlushCache()
            ds = None  # Close file and flush to disk
        return
