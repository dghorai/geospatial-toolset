#!/usr/bin/env python
"""
****************************************************************************
* File Name:      morphometric_analysis.py
* Created:        March 23, 2024
* Author:         Debabrata Ghorai, Ph.D.
* Purpose:        Morphometric Analysis for Prioritizing Sub-watershed and Management
* Description:    Morphometric parameters extraction: linear aspects, areal aspects, relief aspects
* Research Paper: Ghorai, D., Bhunia, G. S., Devulapalli, S., & Shit, P. K. (2022).
                  Morphometric Analysis for Prioritizing Sub-watershed and Management Using Geospatial Technique.
                  Drainage Basin Dynamics: An Introduction to Morphology, Landscape and Modelling, 285-303.
                  Available at: https://link.springer.com/chapter/10.1007/978-3-030-79634-1_13
* Revesions:      N/A
****************************************************************************
"""

import numpy

from osgeo import ogr


class LinearAspects:
    def __init__(self,
                 river_network_shapefile,
                 out_linear_aspects_csvfile,
                 stream_basin_field,
                 stream_order_field,
                 stream_length_field
                 ):
        self.river_network_shapefile = river_network_shapefile
        self.streamBasin = stream_basin_field
        self.streamOrder = stream_order_field
        self.streamLength = stream_length_field
        self.outfile = out_linear_aspects_csvfile

    def _mstrln(self, stream_length, stream_number):
        # Mean Stream Length
        mean_strm_len = stream_length/stream_number
        return mean_strm_len

    def _strlr(self, basin2):
        basin3 = list()
        basin3.append((basin2[0][0], basin2[0][1],
                       basin2[0][2], basin2[0][3], numpy.nan))
        for m in range(len(basin2)-1):
            n1 = basin2[m][3]  # mean_strm_len
            n2 = basin2[m+1][3]  # mean_strm_len
            slr = float(n2)/float(n1)
            basin3.append((basin2[m+1][0], basin2[m+1]
                           [1], basin2[m+1][2], basin2[m+1][3], slr))
        return basin3

    def _bfr(self, basin3):
        # Bifurcation Ratio
        basin4 = list()
        mean_br = list()
        for b in range(len(basin3)-1):
            b1 = basin3[b][1]
            b2 = basin3[b+1][1]
            br = float(b1)/float(b2)
            basin4.append(
                (basin3[b][0], basin3[b][1], basin3[b][2], basin3[b][3], basin3[b][4], br))
            mean_br.append(br)
        basin4.append((basin3[-1][0], basin3[-1][1], basin3[-1]
                       [2], basin3[-1][3], basin3[-1][4], numpy.nan))
        return mean_br, basin4

    def _mbfr(self, mean_br, basin3):
        # Mean Bifurcation Ratio
        rbm = float(sum(mean_br))/(len(basin3)-1)
        return rbm

    def write_linear_aspects(self, basinSet, dataList):
        # Loop over the basin
        w = open(self.outfile, 'w')
        w.writelines(
            'River_Basin,Stream_Order,Stream_Number,Stream_Length,Mean_Length,Length_Ratio,Bifurcation_Ratio,Mean_Bifurcation_Ratio\n')
        basinSet.sort()
        for i in basinSet:
            # Basin/Sub-Basin ID
            print(i)
            basin1 = list()
            rorder1 = list()
            for j in dataList:
                if i == j[0]:
                    basin1.append(j)
                    rorder1.append(j[1])
            rorder2 = list(set(rorder1))
            # Loop over the stream order
            rorder2.sort()
            basin2 = list()
            for ii in rorder2:
                rorder3 = list()
                for jj in basin1:
                    if ii == jj[1]:
                        rorder3.append(jj[2])
                # Stream Number
                stream_number = len(rorder3)
                # Stream Length
                stream_length = sum(rorder3)
                # Mean Stream Length
                mean_strm_len = self._mstrln(stream_length, stream_number)
                basin2.append(
                    (ii, stream_number, stream_length, mean_strm_len))
            # Stream Length Ratio
            basin3 = self._strlr(basin2)
            # Bifurcation Ratio
            mean_br, basin4 = self._bfr(basin3)
            # Mean Bifurcation Ratio
            rbm = self._mbfr(mean_br, basin3)
            # Output CSV file writing
            for f in basin4:
                w.writelines((str(i)+","+",".join(str(ff)
                             for ff in f)+","+str(rbm)+"\n"))
        w.close()
        return

    def extract_linear_aspects(self):
        # get feature layer
        ds = ogr.Open(self.river_network_shapefile)
        layer = ds.GetLayer(0)
        # loop over the features
        basinList = list()
        dataList = list()
        for i in range(layer.GetFeatureCount()):
            feature = layer.GetFeature(i)
            sb = feature.GetField(self.streamBasin)
            so = feature.GetField(self.streamOrder)
            sl = feature.GetField(self.streamLength)
            basinList.append(sb)
            dataList.append((sb, so, sl))
        # Get total basin counts
        basinSet = list(set(basinList))
        # save parameters
        self.write_linear_aspects(basinSet, dataList)
        return


class ArealAspects:
    def __init__(self,
                 drainage_basin_boundary,
                 river_network_shapefile,
                 out_areal_aspects_csvfile,
                 drainage_basinid_field,
                 drainage_basin_area_field,
                 drainage_basin_perimeter_field,
                 drainage_basin_length_field,
                 stream_basin_field,
                 stream_length_field,
                 stream_order_field
                 ):
        self.inBasin = drainage_basin_boundary
        self.inDrainage = river_network_shapefile
        self.basinID = drainage_basinid_field
        self.basinArea = drainage_basin_area_field
        self.basinPerimeter = drainage_basin_perimeter_field
        self.basinLength = drainage_basin_length_field
        self.drainageBasin = stream_basin_field
        self.drainageLength = stream_length_field
        self.drainageOrder = stream_order_field
        self.outfile = out_areal_aspects_csvfile

    def write_areal_aspects(self, basinSeq, dataMerge):
        # Aerial Aspect Calculation
        w = open(self.outfile, 'w')
        w.writelines('BasinID,Parameters,Results\n')
        for i in basinSeq:
            print(i)
            for j in dataMerge:
                if i == j[0][0]:
                    A = float(j[0][1])  # Area
                    P = float(j[0][2])  # Perimeter
                    Lb = float(j[0][3])  # Basin Length
                    Dd = float(j[0][5])/A  # Drainage Density
                    Rt = int(j[0][6])/P  # Texture Ratio
                    Fs = int(j[0][4])/A  # Stream Frequency
                    Ff = A/Lb**2  # Form Factor
                    Re = (2*numpy.sqrt(A/(numpy.pi)))/Lb  # Elongation Ratio
                    Rc = (4*(numpy.pi)*A)/P**2  # Circularity Ratio
                    Lg = (1/Dd)*2  # Length of Overland Flow
                    # ----------------------------------------------
                    w.writelines(str(i)+","+'Area (sq.km.)'+","+str(A)+"\n")
                    w.writelines(""+","+'Perimeter (km.)'+","+str(P)+"\n")
                    w.writelines(""+","+'Basin Length (km.)'+","+str(Lb)+"\n")
                    w.writelines(""+","+'Drainage Density (Dd)' +
                                 ","+str(Dd)+"\n")
                    w.writelines(""+","+'Texture Ratio (Rt)'+","+str(Rt)+"\n")
                    w.writelines(""+","+'Stream Frequency (Fs)' +
                                 ","+str(Fs)+"\n")
                    w.writelines(""+","+'Form Factor (Ff)'+","+str(Ff)+"\n")
                    w.writelines(""+","+'Elongation Ratio (Re)' +
                                 ","+str(Re)+"\n")
                    w.writelines(
                        ""+","+'Circularity Ratio (Rc)'+","+str(Rc)+"\n")
                    w.writelines(
                        ""+","+'Length of Overland Flow (Lg)'+","+str(Lg)+"\n")
                    break
        w.close()
        return

    def extract_areal_aspects(self):
        # Input shapefile layer
        # Basin
        ds_basin = ogr.Open(self.inBasin)
        layer1 = ds_basin.GetLayer(0)
        # Drainage network
        ds_river = ogr.Open(self.inDrainage)
        layer2 = ds_river.GetLayer(0)

        # Loop over the basins
        dataList1 = list()
        for i in range(layer1.GetFeatureCount()):
            feature1 = layer1.GetFeature(i)
            BID = feature1.GetField(self.basinID)
            Ar = feature1.GetField(self.basinArea)
            Pr = feature1.GetField(self.basinPerimeter)
            Ln = feature1.GetField(self.basinLength)
            dataList1.append((BID, Ar, Pr, Ln))

        # Loop over the drainage network
        basinList = list()
        dataList2 = list()
        for i in range(layer2.GetFeatureCount()):
            feature2 = layer2.GetFeature(i)
            DB = feature2.GetField(self.drainageBasin)
            DL = feature2.GetField(self.drainageLength)
            DO = feature2.GetField(self.drainageOrder)
            basinList.append(DB)
            dataList2.append((DB, DL, DO))

        # Loop over the basin list
        basinSet = list(set(basinList))
        dataList3 = list()
        for i in basinSet:
            dl = list()
            so = list()
            for j in dataList2:
                if i == j[0]:
                    dl.append(j[1])
                    so.append(j[2])
            so_f = list()
            for k in so:
                if k == 1:
                    so_f.append(k)
            basin_id = i
            strm_total = len(dl)
            strm_length = sum(dl)
            first_order_tot = len(so_f)
            dataList3.append(
                (basin_id, strm_total, strm_length, first_order_tot))

        # Loop over the dataList
        basinSeq = list()
        dataMerge = list()
        for i in dataList1:
            basinSeq.append(i[0])
            data1 = list()
            for j in dataList3:
                if i[0] == j[0]:
                    # Basin_ID, Basin_Area, Basin_Perimeter, Basin_Length, Total_Streams, Total_Stream_Length, Total_First_Order
                    data1.append((i[0], i[1], i[2], i[3], j[1], j[2], j[3]))
                    break
            dataMerge.append(data1)
        basinSeq.sort()

        # Aerial Aspect Calculation
        self.write_areal_aspects(basinSeq, dataMerge)
        return


class ReliefAspects:
    def __init__(self,
                 drainage_basin_boundary,
                 river_network_shapefile,
                 drainage_basinid_field,
                 drainage_basin_area_field,
                 drainage_basin_perimeter_field,
                 drainage_basin_length_field,
                 drainage_bason_max_elevation_field,
                 drainage_basin_min_elevation_field,
                 stream_basin_field,
                 stream_order_field,
                 stream_length_field,
                 out_relief_aspects_csvfile
                 ):
        self.inBasin = drainage_basin_boundary
        self.inDrainage = river_network_shapefile
        self.basinID = drainage_basinid_field
        self.basinArea = drainage_basin_area_field
        self.basinPerimeter = drainage_basin_perimeter_field
        self.basinLength = drainage_basin_length_field
        self.drainageBasin = stream_basin_field
        self.drainageLength = stream_length_field
        self.drainageOrder = stream_order_field
        self.maxElevation = drainage_bason_max_elevation_field
        self.minElevation = drainage_basin_min_elevation_field
        self.outfile = out_relief_aspects_csvfile

    def write_relief_aspects(self, basinSeq, dataMerge):
        # Aerial Aspect Calculation
        w = open(self.outfile, 'w')
        w.writelines('BasinID,Parameters,Results\n')
        for i in basinSeq:
            print(i)
            for j in dataMerge:
                if i == j[0][0]:
                    A = float(j[0][1])  # Area
                    Lb = float(j[0][3])  # Basin Length
                    Dd = float(j[0][5])/A  # Drainage Density
                    H = float(j[0][7])  # Maximum Height (Meter)
                    h = float(j[0][8])  # Minimum Height (Meter)
                    # --------------------------------------------
                    R = float(H - h)  # Relative Relief (meter)
                    Rh = (R/1000.0)/Lb  # Relief Ratio (km/km)
                    Dis = R/H  # Dissection Index (meter/meter)
                    Rn = Dd*(R/1000.0)  # Ruggedness Number (sq.km. * km)
                    # ----------------------------------------------
                    w.writelines(
                        str(i)+","+'Relative Relief (R)'+","+str(R)+"\n")
                    w.writelines(''+","+'Relief Ratio (Rh)'+","+str(Rh)+"\n")
                    w.writelines(
                        ''+","+'Dissection Index (Dis)'+","+str(Dis)+"\n")
                    w.writelines(
                        ''+","+'Ruggedness Number (Rn)'+","+str(Rn)+"\n")
                    break
        w.close()
        return

    def extract_relief_aspects(self):
        ds_basin = ogr.Open(self.inBasin)
        layer1 = ds_basin.GetLayer(0)
        # Drainage network
        ds_river = ogr.Open(self.inDrainage)
        layer2 = ds_river.GetLayer(0)

        # Loop over the basins
        dataList1 = list()
        for i in range(layer1.GetFeatureCount()):
            feature1 = layer1.GetFeature(i)
            BID = feature1.GetField(self.basinID)
            Ar = feature1.GetField(self.basinArea)
            Pr = feature1.GetField(self.basinPerimeter)
            Ln = feature1.GetField(self.basinLength)
            Mx = feature1.GetField(self.maxElevation)
            Mn = feature1.GetField(self.minElevation)
            dataList1.append((BID, Ar, Pr, Ln, Mx, Mn))
            del i

        # Loop over the drainage network
        basinList = list()
        dataList2 = list()
        for i in range(layer2.GetFeatureCount()):
            feature2 = layer2.GetFeature(i)
            DB = feature2.GetField(self.drainageBasin)
            DL = feature2.GetField(self.drainageLength)
            DO = feature2.GetField(self.drainageOrder)
            basinList.append(DB)
            dataList2.append((DB, DL, DO))
            del i
        # Loop over the basin list
        basinSet = list(set(basinList))
        dataList3 = list()
        for i in basinSet:
            dl = list()
            so = list()
            for j in dataList2:
                if i == j[0]:
                    dl.append(j[1])
                    so.append(j[2])
            so_f = list()
            for k in so:
                if k == 1:
                    so_f.append(k)
            basin_id = i
            strm_total = len(dl)
            strm_length = sum(dl)
            first_order_tot = len(so_f)
            dataList3.append(
                (basin_id, strm_total, strm_length, first_order_tot))

        # Loop over the dataList
        basinSeq = list()
        dataMerge = list()
        for i in dataList1:
            basinSeq.append(i[0])
            data1 = list()
            for j in dataList3:
                if i[0] == j[0]:
                    # Basin_ID, Basin_Area, Basin_Perimeter, Basin_Length, Total_Streams, Total_Stream_Length, Total_First_Order, Max_Elv, Min_Elv
                    data1.append(
                        (i[0], i[1], i[2], i[3], j[1], j[2], j[3], i[4], i[5]))
                    break
            dataMerge.append(data1)
        basinSeq.sort()

        # Aerial Aspect Calculation
        self.write_relief_aspects(basinSeq, dataMerge)
        return


def generate_morphometric_parameters(
    river_network_shapefile,
    drainage_basin_boundary,
    stream_basin_field,
    stream_order_field,
    stream_length_field,
    drainage_basinid_field,
    drainage_basin_area_field,
    drainage_basin_perimeter_field,
    drainage_basin_length_field,
    drainage_bason_max_elevation_field,
    drainage_basin_min_elevation_field,
    out_linear_aspects_csvfile,
    out_areal_aspects_csvfile,
    out_relief_aspects_csvfile
):
    # stream_basin_field = "SUBBASIN"
    # stream_order_field = "DGORD"
    # stream_length_field = "Length_KM"
    # basinID = "COMID"
    # basinArea = "A_SQKM"
    # basinPerimeter = "P_KM"
    # basinLength = "LB_KM"
    # basin_maxElevation = "MAX_ELV"
    # basin_minElevation = "MIN_ELV"
    linear_aspects = LinearAspects(
        river_network_shapefile,
        out_linear_aspects_csvfile,
        stream_basin_field,
        stream_order_field,
        stream_length_field
    )
    linear_aspects.extract_linear_aspects()

    areal_aspects = ArealAspects(
        drainage_basin_boundary,
        river_network_shapefile,
        out_areal_aspects_csvfile,
        drainage_basinid_field,
        drainage_basin_area_field,
        drainage_basin_perimeter_field,
        drainage_basin_length_field,
        stream_basin_field,
        stream_length_field,
        stream_order_field
    )
    areal_aspects.extract_areal_aspects()

    relief_aspects = ReliefAspects(
        drainage_basin_boundary,
        river_network_shapefile,
        drainage_basinid_field,
        drainage_basin_area_field,
        drainage_basin_perimeter_field,
        drainage_basin_length_field,
        drainage_bason_max_elevation_field,
        drainage_basin_min_elevation_field,
        stream_basin_field,
        stream_order_field,
        stream_length_field,
        out_relief_aspects_csvfile
    )
    relief_aspects.extract_relief_aspects()
    return
