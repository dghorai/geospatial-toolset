"""
Created on Sun Apr 24 2022

@author: Debabrata Ghorai, Ph.D.

Merge vector files.

"""

import os

from osgeo import ogr
from config import VectordataConfig
from logger import CustomException


class MergeVectorFiles:
    def __init__(self, vdconfig: VectordataConfig):
        self.vdconfig = vdconfig
        self.outfile = self.vdconfig.vector_outfile
        self.filedir = self.vdconfig.vector_file_dir
        self.geom_type = self.vdconfig.vector_geometry_type

    def find_file_geomtype(self, geom_type=''):
        if geom_type == 'Point':
            geometryType = ogr.wkbPoint
        elif geom_type == 'Line':
            geometryType = ogr.wkbLineString
        elif geom_type == 'Polygon':
            geometryType = ogr.wkbPolygon
        else:
            raise CustomException("Geometry type should be Point/Line/Polygon")
        return geometryType

    def merge_vector_files(self):
        # get list of vector files
        fileList = os.listdir(self.filedir)
        # get spatial reference from first file
        infc = [self.filedir+'/'+f for f in fileList if f.endswith('.shp')][0]
        shapefile = ogr.Open(infc)
        layer = shapefile.GetLayer(0)
        srs = layer.GetSpatialRef()
        srs.ExportToWkt()
        
        # get driver
        out_driver = ogr.GetDriverByName('ESRI Shapefile')
        geometryType = self.find_file_geomtype(geom_type=self.geom_type)
        
        if os.path.exists(self.outfile):
            out_driver.DeleteDataSource(self.outfile)
            out_ds = out_driver.CreateDataSource(self.outfile)
            out_layer = out_ds.CreateLayer(
                "lyr_outfile", 
                srs, 
                geom_type=geometryType
            )
        else:
            out_ds = out_driver.CreateDataSource(self.outfile)
            out_layer = out_ds.CreateLayer(
                "lyr_outfile", 
                srs, 
                geom_type=geometryType
            )
            
        # loop over vector files and merge them
        for f in fileList:
            if f.endswith('.shp'):
                ds = ogr.Open(self.filedir+"/"+f)
                lyr = ds.GetLayer()
                
                if lyr.GetGeomType() == geometryType:
                    for feat in lyr:
                        out_feat = ogr.Feature(out_layer.GetLayerDefn())
                        out_feat.SetGeometry(feat.GetGeometryRef().Clone())
                        out_layer.CreateFeature(out_feat)
                        out_layer.SyncToDisk()
                else:
                    pass
                    
                ds = None
                
        return


def merge_shapefiles(vector_file_dir, vector_outfile, geometry_type=None):
    vdconfig = VectordataConfig(
        vector_outfile=vector_outfile,
        vector_file_dir=vector_file_dir,
        vector_geometry_type=geometry_type
    )
    obj = MergeVectorFiles(vdconfig=vdconfig)
    obj.merge_vector_files()
    return
