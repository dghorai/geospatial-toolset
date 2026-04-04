# -*- coding: utf-8 -*-
"""
Created on Mon Jul  18 20:41:59 2022

@author: Debabrata Ghorai, Ph.D.

Convert vector to raster.
"""


from osgeo import gdal
from osgeo import ogr
#from osgeo import gdalconst


def create_raster(out_rast, source_layer, x_res=100, y_res=100, dtype=gdal.GDT_Float32):
    NoData_value = -9999
    x_min, x_max, y_min, y_max = source_layer.GetExtent()
    cols = int( (x_max - x_min) / x_res )
    rows = int( (y_max - y_min) / y_res )
    raster = gdal.GetDriverByName('GTiff').Create(out_rast, cols, rows, 1, dtype)
    raster.SetGeoTransform((x_min, x_res, 0, y_max, 0, -y_res))
    band = raster.GetRasterBand(1)
    band.SetNoDataValue(NoData_value)
    band.FlushCache()
    return raster


# Filenames for in- and output
in_shp = r"D:\New_Project\Test_data.shp"
out_rast = r"D:\New_Project\Temp Folder\test_rast2.tif"
temp_folder = r"D:\New_Project\Temp Folder"

x_res = 100
y_res = 100


# Open Shapefile
source_ds = ogr.Open(in_shp)
source_layer = source_ds.GetLayer()

# Assign pixel values to pixel
target_ds = create_raster(out_rast, source_layer, x_res, y_res, dtype=gdal.GDT_Float32)

driver = ogr.GetDriverByName('ESRI Shapefile')


for cnt, feat in enumerate(source_layer):
    print(cnt)
    burn_value = float(feat.GetField("Total"))
    fn = temp_folder+'\\temp_'+str(cnt+1)+'.shp'
    datasource = driver.CreateDataSource(fn)
    target_layer = datasource.CreateLayer(fn, source_layer.GetSpatialRef(), geom_type=ogr.wkbPolygon)
    target_layer.CreateFeature(feat)
    gdal.RasterizeLayer(target_ds, [1], target_layer, burn_values=[burn_value], options=["ALL_TOUCHED=TRUE"])
    datasource.Destroy()

#This makes raster to write to disk
target_ds = None
