# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 2016

@author: Debabrata Ghorai, Ph.D.

Clip Raster Data.
"""

import sys

from osgeo import gdal, gdalnumeric, ogr
from logger import CustomException
from config import RasterdataConfig
from src.utils.ref_scripts import world2Pixel
from src.utils.geo_utils import write_geotiff_file, np2gdal_dtype


class ClipRaster:
    """
    Clip or subset raster data.
    """

    def __init__(self, rstconfig: RasterdataConfig):
        """constructure"""
        self.rstconfig = rstconfig
        self.inRaster = self.rstconfig.raster_infile
        self.inPolygon = self.rstconfig.polygon_infile

    def subset_by_extent(self):
        # Open the image as a read only image
        ds = gdal.Open(self.inRaster, gdal.GA_ReadOnly)
        # Check the ds has open successfully
        if ds is None:
            raise CustomException("The raster could not open")
        # get total input raster bands (count)
        inrasband = [band+1 for band in range(ds.RasterCount)]
        # Get image georeferencing information.
        geoMatrix = ds.GetGeoTransform()
        ulX = geoMatrix[0]
        ulY = geoMatrix[3]
        # open shapefile
        shp = ogr.Open(self.inPolygon)
        if len(shp) != 1:
            print("The shapefile with more than 1 record")
            sys.exit(-1)
        source_shp = ogr.GetDriverByName("Memory").CopyDataSource(shp, "")
        # Create an OGR layer from a boundary shapefile
        source_layer = source_shp.GetLayer(0)
        # Convert the layer extent to image pixel coordinates
        minX, maxX, minY, maxY = source_layer.GetExtent()
        ulX, ulY = world2Pixel(geoMatrix, minX, maxY)
        lrX, lrY = world2Pixel(geoMatrix, maxX, minY)
        # Load the source data as a gdalnumeric array
        srcArray = gdalnumeric.LoadFile(self.inRaster)
        if len(inrasband) > 1:
            clip = srcArray[:, ulY:lrY, ulX:lrX]  # For multi-channel raster
        else:
            clip = srcArray[ulY:lrY, ulX:lrX]  # For gray scale raster
        return clip


def clip_raster_by_extent(input_geotiff, clipping_boundary, clipped_geotiff):
    args = RasterdataConfig(
        raster_infile=input_geotiff,
        polygon_infile=clipping_boundary,
        clip_raster_file=clipped_geotiff
    )
    obj = ClipRaster(rstconfig=args)
    clip = obj.subset_by_extent()
    # save raster
    ds = gdal.Open(args.raster_infile, gdal.GA_ReadOnly)
    src_geotransform = ds.GetGeoTransform()
    src_projection = ds.GetProjectionRef()
    ds = None
    write_geotiff_file(
        clip,
        gt=src_geotransform,
        sr=src_projection,
        outfile_path=args.clip_raster_file,
        dtype=np2gdal_dtype(str(clip.dtype))
    )
    return


def clip_raster_by_mask():
    pass
