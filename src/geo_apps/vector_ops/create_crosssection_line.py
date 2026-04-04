# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 2022

@author: Debabrata Ghorai, Ph.D.

Generate perpendicular line along polyline at specific interval and
assign uniqueid to the perpendicular lines.
"""

import os
import math

from osgeo import ogr
from logger import logger
from config import PolylineConfig
from src.utils.ref_scripts import fixed_interval_points
from src.utils.geo_utils import get_shapefile_epsg_code


class XSCL:
    def __init__(self, plconfig: PolylineConfig):
        self.plconfig = plconfig
        self.xscl_available = self.plconfig.is_xscl_available
        self.infc = self.plconfig.inline
        self.outfc = self.plconfig.outxsclline
        self.uidfield = self.plconfig.uidfield
        self.objectidfield = self.plconfig.objectidfield
        self.linefield = self.plconfig.linefield
        self.interval = self.plconfig.interval
        self.offset = self.plconfig.offset

    def construct_perpendicular_line(self, infc, outfc=None, interval=500, offset=500, save=True):
        """Construct perpendicular line"""
        src_epsg_code = get_shapefile_epsg_code(infc)
        if src_epsg_code == 4326:
            # assuming units are in meter
            interval = interval*(1/(111.0*1000))
            offset = offset*(1/(111.0*1000))

        objids = []
        lineids = []
        try:
            # generate perpendicular line nodes
            iPts, iSlp = fixed_interval_points(
                infc, interval, flipline=True, lineslope=True)

            xscl = list()
            for i, j in zip(iPts, iSlp):
                if j == 0.0 or j == -0.0:
                    p1 = i[0], i[1]+offset
                    p2 = i[0], i[1]-offset
                elif j == "Inf":
                    p1 = i[0]+offset, i[1]
                    p2 = i[0]-offset, i[1]
                elif j != 0 and j != "Inf":
                    dx = offset/math.sqrt(1+(1/j**2))
                    dy = offset/(j*math.sqrt(1+(1/j**2)))
                    p1 = i[0]+dx, i[1]-dy
                    p2 = i[0]-dx, i[1]+dy
                else:
                    logger.info("Extra Point:", i, j)
                v1 = p1, p2, i[2]
                xscl.append(v1)

            if save == True:
                # perpendicular lines creation
                shapefile = ogr.Open(infc)
                layer = shapefile.GetLayer(0)
                spr = layer.GetSpatialRef()
                spr.ExportToWkt()
                driver = ogr.GetDriverByName('ESRI Shapefile')
                if os.path.exists(outfc):
                    driver.DeleteDataSource(outfc)
                shapeData = driver.CreateDataSource(outfc)
                lyr = shapeData.CreateLayer(
                    'myLyr2', spr, geom_type=ogr.wkbMultiLineString)
                # create fileds
                lineid = ogr.FieldDefn("LINEID", ogr.OFTInteger)
                lyr.CreateField(lineid)
                objectid = ogr.FieldDefn("OBJECTID", ogr.OFTInteger)
                lyr.CreateField(objectid)
                lyrDef = lyr.GetLayerDefn()
                for ix, row in enumerate(xscl):
                    s, e, n = row[0], row[1], row[2]
                    line = ogr.Geometry(ogr.wkbLineString)
                    line.AddPoint(s[0], s[1])
                    line.AddPoint(e[0], e[1])
                    feature = ogr.Feature(lyrDef)
                    feature.SetGeometry(line)
                    feature.SetField("OBJECTID", ix+1)
                    feature.SetField("LINEID", n)
                    lyr.CreateFeature(feature)
                # Flush
                shapeData.Destroy()
            else:
                for ix, row in enumerate(xscl):
                    n = row[2]
                    objids.append([n, ix+1])
                    lineids.append(n)
        except Exception as e:
            logger.info(e)

        # return
        if save == False:
            return objids, lineids

    def create_xscl_uniqueid(self):
        """Assign uniqueid to perpendicular line"""
        if self.xscl_available:
            if self.outfc:
                objids = []
                lineids = []
                shapefile = ogr.Open(self.outfc)
                layer = shapefile.GetLayer(0)
                for row in range(layer.GetFeatureCount()):
                    feature = layer.GetFeature(row)
                    objid = feature.GetField(self.objectidfield)
                    lineid = feature.GetField(self.linefield)
                    objids.append([lineid, objid])
                    lineids.append(lineid)
            else:
                raise Exception("outfc not provided")
        else:
            self.construct_perpendicular_line(
                infc=self.infc, outfc=self.outfc, interval=self.interval, offset=self.offset)
            objids, lineids = self.construct_perpendicular_line(
                infc=self.infc, interval=self.interval, offset=self.offset, save=False)

        # get unique lineids
        unqlineids = list(set(lineids))
        comid = []
        for i in unqlineids:
            v2 = []
            for j in objids:
                if i == j[0]:
                    v2.append(j[1])
            comid.append([i, v2])

        # write shapefile
        driver = ogr.GetDriverByName('ESRI Shapefile')
        dataSource = driver.Open(self.outfc, 1)
        outlyr = dataSource.GetLayer()
        defn = outlyr.GetLayerDefn()
        if defn.GetFieldIndex(self.uidfield) == -1:
            # create fileds
            outlyr.CreateField(ogr.FieldDefn(self.uidfield, ogr.OFTString))
        # feat_def.AddFieldDefn(uidfield)
        for u in range(outlyr.GetFeatureCount()):
            feat = outlyr.GetFeature(u)
            objid = feat.GetField(self.objectidfield)
            for k in comid:
                for l in range(len(k[1])):
                    if objid == k[1][l]:
                        new_id = str(k[0])+"_"+str(l)
                        feat.SetField(self.uidfield, new_id)
                        outlyr.SetFeature(feat)
        # Flush
        dataSource.Destroy()
        return


def generate_river_xscl(
    river_network_file,
    out_xscl_file,
    xscl_uid_field,
    xscl_interval,
    xscl_length,
    is_xscl_exist=False,
    existing_xscl_objectid=None,
    existing_xscl_lineid=None
):
    plconfig = PolylineConfig(
        is_xscl_available=is_xscl_exist,
        inline=river_network_file,
        outxsclline=out_xscl_file,
        uidfield=xscl_uid_field,
        objectidfield=existing_xscl_objectid,
        linefield=existing_xscl_lineid,
        interval=xscl_interval,
        offset=xscl_length
    )
    obj = XSCL(plconfig=plconfig)
    obj.create_xscl_uniqueid()
    return
