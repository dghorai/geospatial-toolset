# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 18:41:31 2024

@author: Debabrata Ghorai, Ph.D.

Distance measurement from baseline to past shorelines.

"""

from osgeo import ogr
from src.utils.geo_utils import dist_calc


# historical shoreline's distance measurement from baseline
def distance_from_baseline(qset, alist):
    dists = []
    for seq in qset:
        tlist = []
        for f in alist:
            if seq == f[0]:
                tlist.append(f[1:])
        base = None
        for t in tlist:
            if t[1] == 'Baseline':
                base = t[3]
        cnt = 1
        for t2 in tlist:
            if t2[1] != 'Baseline':
                dist = dist_calc(base, t2[3])
                dists.append((seq, cnt, t2[0], dist, t2[2]))
            cnt += 1
    return dists


def write_historical_shoreline_distance(dists, out_csvfile):
    # write csv file
    w = open(out_csvfile, 'w')
    w.writelines('TrID,YearID,Date,Distance,E\n')
    for ii in dists:
        w.writelines(",".join(str(i) for i in ii)+"\n")
    w.close()
    return


def extract_distance_parameters(layer, fields):
    qlist = []
    alist = []
    for i in range(layer.GetFeatureCount()):
        feat = layer.GetFeature(i)
        # get seqid field
        seqid = feat.GetField(fields[0])
        # get date field
        date = feat.GetField(fields[1])
        # get E field
        e = feat.GetField(fields[2])
        # get name field
        name = feat.GetField(fields[3])
        # get geometry
        geom = feat.GetGeometryRef()
        point = geom.GetPoints()
        # append values
        alist.append((seqid, date, name, e, point[0]))
        qlist.append(seqid)
    return qlist, alist


def measure_historical_shoreline_distance(shore_transect_xpoint_file, out_csvfile):
    """
    Historical shoreline's distance measurement from baseline.

    Required Datasets:
    A merged point file of the following layers:
    - Baseline and Transect intersects points
    - Historical shorelines and Transect intersect points

    Required Fields:
    a) SeqID (Index 1)
    b) DATE (Index 2)
    c) E (Index 3)
    d) NAME (Index 4)
    """

    # data source
    ds = ogr.Open(shore_transect_xpoint_file)
    layer = ds.GetLayer(0)
    layerDefn = layer.GetLayerDefn()

    # get spatial reference
    sr = layer.GetSpatialRef()
    sr.ExportToWkt()

    # get all the fields
    fields = []
    for i in range(layerDefn.GetFieldCount()):
        field_name = layerDefn.GetFieldDefn(i).GetName()
        fields.append(field_name)

    # calculate historical distances
    try:
        # see the total no. of fields should be four
        if len(fields) == 4:
            qlist, alist = extract_distance_parameters(layer, fields)
            # calculate unique seqid
            qset = list(set(qlist))
            # calculate distances
            dists = distance_from_baseline(qset, alist)
            # write csv file
            write_historical_shoreline_distance(dists, out_csvfile)
    except:
        raise Exception("Check the input datasets and it's attributes")
    return
