# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 2022

@author: Debabrata Ghorai, Ph.D.

Calculate cumulative drainage area.

"""

from osgeo import ogr
from logger import logger
from config import PolylineConfig, PolygonConfig


class DrainageAreaCalc:
    def __init__(self, plconfig: PolylineConfig, pgconfig: PolygonConfig):
        self.plconfig = plconfig
        self.pgconfig = pgconfig
        self.inline = self.plconfig.inline
        self.inpolygon = self.pgconfig.inpolygon
        self.fnodeid = self.plconfig.fnodeid
        self.tnodeid = self.plconfig.tnodeid
        self.lineid = self.plconfig.lineid
        self.seqid = self.plconfig.seqid
        self.cumfield_name = self.pgconfig.cumfield_name

    def get_polygon_area(self, seqdict, tnodedict, areadict):
        logger.info(seqdict)
        newarea = {}
        for s2 in seqdict:
            if s2 in areadict:
                nArea = areadict[s2]
            if s2 in tnodedict:
                a1 = []
                for s3 in tnodedict[s2]:
                    v1 = areadict[s3][0]
                    a1.append(v1)
                v2 = nArea+a1
                v3 = sum(v2)
                if not s2 in newarea:
                    newarea[s2] = []
                newarea[s2].append(v3)
        return newarea

    def iter_polygon_area(self, seqdict, tnodedict, areadict):
        na = self.get_polygon_area(seqdict, tnodedict, areadict)
        for k1 in areadict.keys():
            for k2 in na.keys():
                if k1 == k2:
                    areadict[k1] = na[k2]
        return areadict

    def cumulative_drainage_area(self):
        """calculate drainage area"""
        seqdict = {}
        tnodedict = {}
        rShapefile = ogr.Open(self.inline)
        rLayer = rShapefile.GetLayer(0)
        for i in range(rLayer.GetFeatureCount()):
            feature = rLayer.GetFeature(i)
            fnode = feature.GetField(self.fnodeid)
            tnode = feature.GetField(self.tnodeid)
            line = feature.GetField(self.lineid)
            seqs = feature.GetField(self.seqid)
            # dict
            if not seqs in seqdict:
                seqdict[seqs] = []
            seqdict[seqs].append(line)
            # dict
            if not tnode in tnodedict:
                tnodedict[tnode] = []
            tnodedict[tnode].append(fnode)

        # read polygon file
        areadict = {}
        tShapefile = ogr.Open(self.inpolygon)
        tLayer = tShapefile.GetLayer(0)
        for j in range(tLayer.GetFeatureCount()):
            feat = tLayer.GetFeature(j)
            plyid = feat.GetField(self.lineid)
            geom = feat.GetGeometryRef()
            area = geom.GetArea()
            if not plyid in areadict:
                areadict[plyid] = []
            areadict[plyid].append(area)

        # calc cumulative area
        temp_area = []
        for k in seqdict.keys():
            orderid = seqdict[k]
            next_area = self.get_polygon_area(orderid, tnodedict, areadict)
            next_dict = self.iter_polygon_area(areadict, tnodedict, next_area)
            areadict = next_dict
            temp_area.append(next_dict)

        # return final cumulative area
        newAreaDict = temp_area[-1]

        # update field
        driver = ogr.GetDriverByName('ESRI Shapefile')
        dataSource = driver.Open(self.inpolygon, 1)
        lyr = dataSource.GetLayer()
        # create fileds
        cumfield = ogr.FieldDefn(self.cumfield_name, ogr.OFTReal)
        lyr.CreateField(cumfield)
        for u in range(lyr.GetFeatureCount()):
            feat = lyr.GetFeature(u)
            obj = feat.GetField(self.lineid)
            if obj in newAreaDict:
                feat.SetField(self.cumfield_name, newAreaDict[obj][0])
                lyr.SetFeature(feat)
        # flash
        dataSource.Destroy()


def get_cumulative_drainage_area(
    drainage_network_file,
    fnode_field,
    tnode_field,
    line_objectid_field,
    line_sequence_field,
    catchment_boundary_file,
    drainage_area_field
):
    plconfig = PolylineConfig(
        inline=drainage_network_file,
        fnodeid=fnode_field,
        tnodeid=tnode_field,
        lineid=line_objectid_field,
        seqid=line_sequence_field
    )
    pgconfig = PolygonConfig(
        inpolygon=catchment_boundary_file,
        cumfield_name=drainage_area_field
    )
    obj = DrainageAreaCalc(plconfig=plconfig, pgconfig=pgconfig)
    obj.cumulative_drainage_area()
    return
