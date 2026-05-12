"""
Created on Sun Apr 24 2022

@author: Debabrata Ghorai, Ph.D.

Convert grid center points to square polygon/grid.

"""


import sys

from osgeo import ogr
from logger import logger
from config import PointConfig, PolygonConfig
from src.utils.geo_utils import get_shapefile_epsg_code
from consts import DEG_TO_KM


class CreateGrid:
    def __init__(self, pntconfig: PointConfig, pgconfig: PolygonConfig):
        self.pntconfig = pntconfig
        self.pgconfig = pgconfig
        self.pointfile = self.pntconfig.inpoint
        self.outfile = self.pgconfig.out_grid_polygon
        self.offset = self.pntconfig.point_offset

    def create_grid_nodes(self, layer, src_epsg_code, offset=None):
        if src_epsg_code == 4326:
            # assuming units are in meter
            offset = offset*(1/(DEG_TO_KM*1000))

        xylist = []
        for row in range(layer.GetFeatureCount()):
            feature = layer.GetFeature(row)
            geometry = feature.GetGeometryRef()
            xylist.append(geometry.GetPoints()[0])

        # polygon's four corner point generation
        polCoord = []
        if offset > 0:
            iDis = offset/2.0
            for i in xylist:
                cx, cy = float(i[0]), float(i[1])
                xyPol = [[cx+iDis, cy+iDis], [cx-iDis, cy+iDis],
                         [cx-iDis, cy-iDis], [cx+iDis, cy-iDis]]
                polCoord.append(xyPol)
        else:
            raise Exception("Grid offset not provided")
            
        return polCoord

    def point_to_grid(self):
        # open the data source
        dataSource = ogr.Open(self.pointfile)
        src_epsg_code = get_shapefile_epsg_code(self.pointfile)
        
        if dataSource is None:
            logger.info('Could not open {}'.format(self.pointfile))
            sys.exit(1)  # exit with an error code
            
        # get the data layer
        layer = dataSource.GetLayer()
        polCoord = self.create_grid_nodes(
            layer, 
            src_epsg_code, 
            offset=self.offset
        )
        # create the output shapefile in the above directory
        driver = ogr.GetDriverByName('ESRI Shapefile')
        outDataSource = driver.CreateDataSource(self.outfile)
        outLayer = outDataSource.CreateLayer(
            self.outfile, 
            geom_type=ogr.wkbPolygon
        )
        # create field
        outField = ogr.FieldDefn('OBJECTID', ogr.OFTInteger)
        outField.SetWidth(10)
        # add the field to the shapefile
        outLayer.CreateField(outField)
        # get the feature definition for the shapefile
        featureDefn = outLayer.GetLayerDefn()
        
        # polygon construction
        for ix, feat in enumerate(polCoord):
            ring = ogr.Geometry(ogr.wkbLinearRing)
            poly = ogr.Geometry(ogr.wkbPolygon)
            
            for n in feat:
                ring.AddPoint(n[0], n[1])
                
            ring.CloseRings()
            poly.AddGeometry(ring)
            
            # create new features and set geometry and attribute
            outFeature = ogr.Feature(featureDefn)
            outFeature.SetGeometry(poly)
            outFeature.SetField('OBJECTID', ix+1)
            
            # add new feature to output layer
            outLayer.CreateFeature(outFeature)
            
        # Destroy data source
        outDataSource.Destroy()
        # create *.prj file
        
        try:
            outSpatialRef = layer.GetSpatialRef()
            outSpatialRef.ExportToWkt()
            outSpatialRef.MorphFromESRI()
            file = open(self.outfile.replace(".shp", ".prj"), 'w')
            file.write(outSpatialRef.ExportToWkt())
            file.close()
        except:
            logger.info("Input data has no projection!")
            
        # destry input dataSource (do not destry before creation of projection file)
        dataSource.Destroy()
        
        return


def generate_grid_boundary(in_point_file, point_distance, out_grid_boundary):
    pntconfig = PointConfig(
        inpoint=in_point_file,
        point_offset=point_distance
    )
    pgconfig = PolygonConfig(
        out_grid_polygon=out_grid_boundary
    )
    obj = CreateGrid(pntconfig=pntconfig, pgconfig=pgconfig)
    obj.point_to_grid()
    return
