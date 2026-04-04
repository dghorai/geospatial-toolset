# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 2022

@author: Debabrata Ghorai, Ph.D.

Polyline geometry operation for flood GIS utilities.
"""

from collections import Counter
from itertools import groupby
from operator import itemgetter
from osgeo import ogr
from src.utils.geo_utils import unique_and_newfield, reading_polyline, line_fnt_nodes


class PolylineGeomOps:
    def __init__(self):
        pass

    def segment_link(self, toconnect, fntnodeids):
        """Get line connected id"""
        connlist = []
        for x in toconnect:
            for y in fntnodeids:
                if x[-1] == y[2]:
                    connlist.append([x[0], y[1]])
        return connlist

    def group_segments(self, ftnodes, segment_counts):
        """Group all the segments connected to each other"""
        toconn = Counter([i[2] for i in ftnodes])
        terminals = []
        for j in range(len(toconn)):
            ids, num = list(toconn.keys()), list(toconn.values())
            if num[j] == 1:
                a = ids[j]
                for k in ftnodes:
                    if a == k[2]:
                        terminals.append(k[1])
        # get line segment of the first terminal
        tcid = []
        toconnect = []
        for t in terminals:
            for i in ftnodes:
                if t == i[2]:
                    toconnect.append([t, i[1]])
                    tcid.append(t)
        group_1 = list(set(terminals)-set(list(set(tcid))))
        # get a copy of toconnect
        toconnect2 = toconnect
        # findings all connected line segments
        toconnect3 = []
        for x in range(len(segment_counts)):
            xt = self.segment_link(toconnect, ftnodes)
            toconnect = xt
            if xt == []:
                break
            else:
                toconnect3.append(xt)
        # iCon list normalization
        toconnect4 = []
        for p in toconnect3:
            for q in p:
                toconnect4.append(q)
        # Concatenation previous line ID and next all IDs
        toconnect5 = toconnect4 + toconnect2
        # Combine connected lines ID based on Terminals ID
        group_2 = [(k, [v[1] for v in g])
                   for k, g in groupby(sorted(toconnect5), key=itemgetter(0))]
        # return segment ids
        return group_1, group_2

    def update_field(self, group_1, group_2, layer, feat, comid, fieldname):
        """Update field"""
        for s in group_1:
            if comid == s:
                feat.SetField(fieldname, s)
                layer.SetFeature(feat)
            del s
        for v in group_2:
            if comid == v[0]:
                feat.SetField(fieldname, v[0])
                layer.SetFeature(feat)
            for w in v[1]:
                if comid == w:
                    feat.SetField(fieldname, v[0])
                    layer.SetFeature(feat)
        return

    def find_line_terminals(self, tmnl2, mfl2):
        """Find line segment terminals"""
        tmnllist = list()
        if tmnl2:
            for j in tmnl2:
                for i in mfl2:
                    if j[1] == i[2]:
                        tmnllist.append((i[0], i[1]))
        else:
            raise Exception("Not able to find line segment terminal.")
        return tmnllist

    def generate_hydroid(self, gdisc):
        """Generate river line hydroid"""
        # loop over group id
        hydroids = []
        for k in gdisc.keys():
            segments = gdisc[k]
            # get all the end nodes
            tlist = []
            for i in segments:
                enode = i[2]
                for j in segments:
                    if enode == j[2]:
                        tlist.append((enode, j[0], j[1]))
            # group to-nodes
            count = Counter(tlist)
            # filter to-nodes if count is 1
            tmnl = []
            for t in count.keys():
                if count[t] == 1:
                    tmnl.append((t[1], t[2]))
            # take the first to-node from the above list as starting point of hydro id generation
            newid = [(tmnl[0][0], 1)]
            # loop over segments, find terminals, and group segments
            hydronum = {}
            for ii in range(len(segments)):
                tt = self.find_line_terminals(tmnl, segments)
                tmnl = tt
                if tt:
                    for jj in tt:
                        if not (ii+1) in hydronum:
                            hydronum[ii+1] = []
                        hydronum[ii+1].append(jj[0])
                else:
                    break
            # assign numbers to each group-segment id
            num = 2
            for k in hydronum.keys():
                for v in hydronum[k]:
                    newid.append((v, num))
                    num += 1
            hydroids.append(newid)
        return hydroids

    def line_direction_topology(self, ftnodes):
        """Function to find line direction error"""
        # Findings to_node error
        fNode = []
        tNode = []
        for ft in ftnodes:
            fNode.append(ft[1])
            tNode.append(ft[2])
        # group from/to nodes
        fCnt = Counter(fNode)
        tCnt = Counter(tNode)
        # create from-node list
        fCount = []
        for i, j in zip(fCnt.keys(), fCnt.values()):
            fval = i, j
            fCount.append(fval)
        # create to-node list
        tCount = []
        for k, v in zip(tCnt.keys(), tCnt.values()):
            tval = k, v
            tCount.append(tval)
        # geom operation
        tfMatch = []
        for t in tCount:
            for f in fCount:
                if t[0] == f[0]:
                    tfMatch.append(t)
        # create unique value list
        tfNotMatch = list(set(tCount)-set(tfMatch))
        tfIDs = []
        for tf in tfNotMatch:
            if tf[1] > 1:
                for ft in ftnodes:
                    if tf[0] == ft[2]:
                        tfIDs.append(ft[1])
        return tfIDs

    def CreateObjectID(self, inshp, fieldname="OBJECTID"):
        """Assign OBJECTID to the line segment"""
        driver = ogr.GetDriverByName('ESRI Shapefile')
        # 1 for read and write shapeline
        dataSource = driver.Open(inshp, 1)
        layer = dataSource.GetLayer()
        unqfieldname = unique_and_newfield(layer, fieldname)
        if unqfieldname:
            print("{} already exists".format(fieldname))
        else:
            # update segments
            for u in range(layer.GetFeatureCount()):
                objectid = u+1
                feature = layer.GetFeature(u)
                feature.SetField(fieldname, objectid)
                layer.SetFeature(feature)
            # flash
            dataSource.Destroy()
        return

    def FnTnodeID(self, inshp, isterminal=False, updatefield=True):
        """Generate from-node and to-node id for each line segment"""
        segment_counts, line_nodes = reading_polyline(inshp)
        nodeids, terminal = line_fnt_nodes(line_nodes, segment_counts)
        if updatefield == True:
            # update shapefile
            driver = ogr.GetDriverByName('ESRI Shapefile')
            dataSource = driver.Open(inshp, 1)
            layer = dataSource.GetLayer()
            unqfieldname = unique_and_newfield(layer, "FNODE")
            _ = unique_and_newfield(layer, "TNODE")
            # update segments
            if unqfieldname:
                for u in range(layer.GetFeatureCount()):
                    feature = layer.GetFeature(u)
                    objectid = feature.GetField(unqfieldname)
                    for n in nodeids:
                        if objectid == n[0]:
                            feature.SetField("FNODE", n[1])
                            feature.SetField("TNODE", n[2])
                            layer.SetFeature(feature)
            else:
                for u in range(layer.GetFeatureCount()):
                    objectid = u+1
                    feature = layer.GetFeature(u)
                    for n in nodeids:
                        if objectid == n[0]:
                            feature.SetField("FNODE", n[1])
                            feature.SetField("TNODE", n[2])
                            layer.SetFeature(feature)
            # flash
            dataSource.Destroy()

        # return final nodeids
        if isterminal == True:
            return nodeids, terminal
        else:
            return nodeids

    def CreateGroupID(self, inlineshp, fieldname="GROUPID"):
        """Generate group id for connected lines"""
        driver = ogr.GetDriverByName('ESRI Shapefile')
        dataSource = driver.Open(inlineshp, 1)
        layer = dataSource.GetLayer()
        # shapefile attribute updating
        segment_counts, _ = reading_polyline(inlineshp)
        ftnodes = self.FnTnodeID(inlineshp, updatefield=False)
        group_1, group_2 = self.group_segments(ftnodes, segment_counts)
        unqfieldname = unique_and_newfield(layer, fieldname)
        if unqfieldname:
            for u in range(layer.GetFeatureCount()):
                feat = layer.GetFeature(u)
                comid = feat.GetField(unqfieldname)
                self.update_field(group_1, group_2, layer,
                                  feat, comid, fieldname)
        else:
            for u in range(layer.GetFeatureCount()):
                feat = layer.GetFeature(u)
                comid = u+1
                self.update_field(group_1, group_2, layer,
                                  feat, comid, fieldname)
        # flash
        dataSource.Destroy()
        return "Process Completed!"

    def GenerateHydroID(self, infc, groupid="GROUPID", outfield="HydroID"):
        """Update field"""
        # reading shapefile
        driver = ogr.GetDriverByName("ESRI Shapefile")
        dataSource = driver.Open(infc, 1)
        layer = dataSource.GetLayer()
        # extract segment nodes and hydroids
        gdisc = reading_polyline(infc, fieldname=groupid)
        unqfieldname = unique_and_newfield(layer, outfield)
        hydroids = self.generate_hydroid(gdisc)
        # update segments
        if unqfieldname:
            hydid = 1
            for f in hydroids:
                for ff in f:
                    for i in range(layer.GetFeatureCount()):
                        ft = layer.GetFeature(i)
                        obj = ft.GetField(unqfieldname)
                        if ff[0] == obj:
                            ft.SetField(outfield, hydid)
                            layer.SetFeature(ft)
                            hydid += 1
        else:
            hydid = 1
            for f in hydroids:
                for ff in f:
                    for i in range(layer.GetFeatureCount()):
                        ft = layer.GetFeature(i)
                        obj = i+1
                        if ff[0] == obj:
                            ft.SetField(outfield, hydid)
                            layer.SetFeature(ft)
                            hydid += 1
        # flash
        dataSource.Destroy()
        return "Process Completed!"

    def LineDirectionError(self, infc):
        """Get all line directional error"""
        # streamline nodes accessing
        objects, nodes = reading_polyline(infc)
        tonodes1 = self.FnTnodeID(infc)
        # flip line
        revered_nodes = []
        rsegment_counts = []
        for x, y in zip(nodes, objects):
            ln = []
            for r in reversed(x):
                ln.append(r)
            revered_nodes.append(ln)
            rsegment_counts.append(y)
        # ftnodes of flip lines
        tonodes2, _ = line_fnt_nodes(revered_nodes, rsegment_counts)
        # line dirrectional error
        from_direction_topology = self.line_direction_topology(tonodes2)
        to_direction_topology = self.line_direction_topology(tonodes1)
        direction_errors = from_direction_topology + to_direction_topology
        return direction_errors


class PolygonGeomOps:
    def __init__(self):
        pass


class PointGeomOps:
    def __init__(self):
        pass
