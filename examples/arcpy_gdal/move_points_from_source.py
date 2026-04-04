# -*- coding: utf-8 -*-

# Created on: 2018-07-11 14:15:31.00000
# Author: Debabrata Ghorai
# Usage: Location Point; Waterbody Polygon; Post-Code Boundary Polygon
# Description: This script will move all the location points which are fall inside of waterbody polygon within post-code boundary


# Import arcpy module
import os
import time
import arcpy

from shapely.geometry import Point, MultiPoint, Polygon
from shapely.ops import nearest_points

# config env
arcpy.env.overwriteOutput = True

print("Start : %s" % time.ctime())
t0 = time.time()

# Script arguments
LocationPoint = "D:\\Work\\Location_Points.shp" # provide a default value if unspecified
WaterbodyPolygon = "D:\\Work\\Waterbody_Polygon.shp" # provide a default value if unspecified
PostCodePolygon = "D:\\Work\\Postcode_Polygon.shp" # provide a default value if unspecified
Folder = "D:\\Work\\Processing" # provide a default value if unspecified
OutputPoints = "D:\\Work\\Processing\\new_point.shp"

# Some functions
def GetNearPoint(poly_list, plon, plat):
    orig = Point(plon, plat)
    dstns = list()
    for i in poly_list:
        pnt_data = Point(i[0], i[1])
        dstns.append(pnt_data)
    destinations = MultiPoint(dstns)
    nearest_coords = nearest_points(orig, destinations)
    xy = nearest_coords[1].x, nearest_coords[1].y
    return xy

if os.path.basename(LocationPoint).split(".")[1] == "shp":
    if arcpy.Exists(OutputPoints):
        arcpy.Delete_management(OutputPoints, "")

    LocationPoint_Layer = "LocationPoint_Layer"
    PostCodePolygon_Lyr2 = "PostCodePolygon_Lyr2"
    PostCodePolygon_Layer = "PostCodePolygon_Layer"
    PostcodeWaterbodyUnion_Layer = "PostcodeWaterbodyUnion_Layer"
    Location_Points_CopyFeatures_Lyr = "Location_Points_CopyFeatures_Lyr"
    Location_Points_CopyFeatures = Folder+"\\Location_Points_CopyFeatures.shp"
    PostCodePolygon_Copy = Folder+"\\PostCodePolygon_Copy.shp"
    Waterbody_Polygon_Buffer = Folder+"\\Waterbody_Polygon_Buffer.shp"
    Waterbody_Polygon_Buffer_Dis = Folder+"\\Waterbody_Polygon_Buffer_Dis.shp"
    Waterbody_Polygon_Buffer_Fnl = Folder+"\\Waterbody_Polygon_Buffer_Fnl.shp"
    Postcode_Waterbody_Union = Folder+"\\Postcode_Waterbody_Union.shp"

    print("Location Points Within Waterbody")
    # Process: Make Feature Layer
    arcpy.MakeFeatureLayer_management(LocationPoint, LocationPoint_Layer, "", "", "")
    # Process: Select Layer By Location
    arcpy.SelectLayerByLocation_management(LocationPoint_Layer, "INTERSECT", WaterbodyPolygon, "", "NEW_SELECTION", "NOT_INVERT")
    # Process: Copy Features
    arcpy.CopyFeatures_management(LocationPoint_Layer, Location_Points_CopyFeatures, "", "0", "0", "0")
    print("Postcode Boundary for Location Points")
    arcpy.MakeFeatureLayer_management(PostCodePolygon, PostCodePolygon_Lyr2, "", "", "")
    arcpy.SelectLayerByLocation_management(PostCodePolygon_Lyr2, "INTERSECT", Location_Points_CopyFeatures, "", "NEW_SELECTION", "NOT_INVERT")
    arcpy.CopyFeatures_management(PostCodePolygon_Lyr2, PostCodePolygon_Copy, "", "0", "0", "0")
    print("Union Waterbody and Postcode Polygon")
    # Process: Buffer
    arcpy.Buffer_analysis(WaterbodyPolygon, Waterbody_Polygon_Buffer, "0.0045 DecimalDegrees", "FULL", "ROUND", "NONE", "", "PLANAR")
    # Process: Dissolve
    arcpy.Dissolve_management(Waterbody_Polygon_Buffer, Waterbody_Polygon_Buffer_Dis, "", "", "MULTI_PART", "DISSOLVE_LINES")
    # Process: Multipart To Singlepart
    arcpy.MultipartToSinglepart_management(Waterbody_Polygon_Buffer_Dis, Waterbody_Polygon_Buffer_Fnl)
    # Process: Add Field (2)
    arcpy.AddField_management(Waterbody_Polygon_Buffer_Fnl, "WID", "LONG", "", "", "", "", "NULLABLE", "NON_REQUIRED", "")
    # Process: Calculate Field
    arcpy.CalculateField_management(Waterbody_Polygon_Buffer_Fnl, "WID", "2", "PYTHON_9.3", "")
    # Process: Add Field
    arcpy.AddField_management(PostCodePolygon_Copy, "PID", "LONG", "", "", "", "", "NULLABLE", "NON_REQUIRED", "")
    # Process: Calculate Field (2)
    arcpy.CalculateField_management(PostCodePolygon_Copy, "PID", "1", "PYTHON_9.3", "")
    # Process: Union
    arcpy.Union_analysis(str(Waterbody_Polygon_Buffer_Fnl)+" #;"+str(PostCodePolygon_Copy)+" #", Postcode_Waterbody_Union, "ALL", "", "GAPS")
    arcpy.Delete_management(Waterbody_Polygon_Buffer, "")
    arcpy.Delete_management(Waterbody_Polygon_Buffer_Dis, "")
    arcpy.Delete_management(Waterbody_Polygon_Buffer_Fnl, "")
    print("Move Location Points")
    # Loop over postcode boundary
    ptList = list()
    arcpy.MakeFeatureLayer_management(PostCodePolygon_Copy, PostCodePolygon_Layer, "", "", "")
    arcpy.MakeFeatureLayer_management(Postcode_Waterbody_Union, PostcodeWaterbodyUnion_Layer, "", "", "")
    arcpy.MakeFeatureLayer_management(Location_Points_CopyFeatures, Location_Points_CopyFeatures_Lyr, "", "", "")
    rows = arcpy.da.SearchCursor(PostCodePolygon_Copy, ["FID", "PID"])
    for row in rows:
        try:
            print("\tPostCode ObjectID : "+str(row[0]))
            PostCode_Polygon_CopyFeatures = Folder+"\\PostCode_"+str(row[0])+".shp"
            PostCode_Negative_Buffer = Folder+"\\PostCode_Negative_Buffer"+str(row[0])+".shp"
            PostcodeWaterbodyUnion_CopyFeatures = Folder+"\\WaterBodyTemp1_"+str(row[0])+".shp"
            PostcodeWaterbodyUnion_CopyFeatures2 = Folder+"\\WaterBodyTemp2_"+str(row[0])+".shp"
            PostcodeWaterbodyUnion_CopyFeatures_Sig = Folder+"\\WaterBody_"+str(row[0])+".shp"
            PostcodeWaterbodyUnion_CopyFeatures_Sig2 = Folder+"\\NearBoundary_"+str(row[0])+".shp"
            PostcodeWaterbodyPoints_CopyFeatures = Folder+"\\LocPoints_"+str(row[0])+".shp"
            arcpy.SelectLayerByAttribute_management(PostCodePolygon_Layer, "NEW_SELECTION", "\"FID\" = "+str(row[0]))
            arcpy.CopyFeatures_management(PostCodePolygon_Layer, PostCode_Polygon_CopyFeatures, "", "0", "0", "0")
            arcpy.Buffer_analysis(PostCode_Polygon_CopyFeatures, PostCode_Negative_Buffer, "-0.000045 DecimalDegrees", "FULL", "ROUND", "NONE", "", "PLANAR")
            # Waterbody selection within postcode row
            arcpy.SelectLayerByLocation_management(PostcodeWaterbodyUnion_Layer, "INTERSECT", PostCode_Negative_Buffer, "", "NEW_SELECTION", "NOT_INVERT")
            arcpy.SelectLayerByAttribute_management(PostcodeWaterbodyUnion_Layer, "SUBSET_SELECTION", "\"WID\" = 2")
            arcpy.CopyFeatures_management(PostcodeWaterbodyUnion_Layer, PostcodeWaterbodyUnion_CopyFeatures, "", "0", "0", "0")
            row4 = arcpy.da.SearchCursor(PostcodeWaterbodyUnion_CopyFeatures, ["SHAPE@"])
            result2 = None
            for row5 in row4:
                geometry1 = row5[0]
                if geometry1.isMultipart == True:
                    result2 = "Multipart"
                else:
                    result2 = "Singlepart"
            if result2 == "Multipart":
                arcpy.MultipartToSinglepart_management(PostcodeWaterbodyUnion_CopyFeatures, PostcodeWaterbodyUnion_CopyFeatures_Sig)
            else:
                arcpy.CopyFeatures_management(PostcodeWaterbodyUnion_CopyFeatures, PostcodeWaterbodyUnion_CopyFeatures_Sig, "", "0", "0", "0")
            arcpy.Delete_management(PostcodeWaterbodyUnion_CopyFeatures, "")
            # PostCode selection without waterbody
            arcpy.SelectLayerByLocation_management(PostcodeWaterbodyUnion_Layer, "INTERSECT", PostCode_Negative_Buffer, "", "NEW_SELECTION", "NOT_INVERT")
            arcpy.SelectLayerByAttribute_management(PostcodeWaterbodyUnion_Layer, "SUBSET_SELECTION", "\"WID\" = 0")
            arcpy.CopyFeatures_management(PostcodeWaterbodyUnion_Layer, PostcodeWaterbodyUnion_CopyFeatures2, "", "0", "0", "0")
            row2 = arcpy.da.SearchCursor(PostcodeWaterbodyUnion_CopyFeatures2, ["SHAPE@"])
            result = None
            for row3 in row2:
                geometry = row3[0]
                if geometry.isMultipart == True:
                    result = "Multipart"
                else:
                    result = "Singlepart"
            if result == "Multipart":
                arcpy.MultipartToSinglepart_management(PostcodeWaterbodyUnion_CopyFeatures2, PostcodeWaterbodyUnion_CopyFeatures_Sig2)
            else:
                arcpy.CopyFeatures_management(PostcodeWaterbodyUnion_CopyFeatures2, PostcodeWaterbodyUnion_CopyFeatures_Sig2, "", "0", "0", "0")
            arcpy.Delete_management(PostcodeWaterbodyUnion_CopyFeatures2, "")
            arcpy.Delete_management(PostCode_Negative_Buffer, "")
            # Points selection within selected waterbody polygon
            arcpy.SelectLayerByLocation_management(Location_Points_CopyFeatures_Lyr, "INTERSECT", PostcodeWaterbodyUnion_CopyFeatures_Sig, "", "NEW_SELECTION", "NOT_INVERT")
            arcpy.CopyFeatures_management(Location_Points_CopyFeatures_Lyr, PostcodeWaterbodyPoints_CopyFeatures, "", "0", "0", "0")
            # Get postcode nodes list
            curs_poly = arcpy.da.SearchCursor(PostcodeWaterbodyUnion_CopyFeatures_Sig2, ["SHAPE@", "FID"])
            tot_pnt_list = list()
            for cur in curs_poly:
                partnum = 0
                for part in cur[0]:
                    for pnt in part:
                        if pnt:
                            tot_pnt_list.append((pnt.X, pnt.Y))
                        else:
                            print ("\t\tInterior Ring:")
                    partnum += 1
                del cur
            # Get closest point
            curs_pnts = arcpy.da.SearchCursor(PostcodeWaterbodyPoints_CopyFeatures, ["SHAPE@", "FID"])
            for cur in curs_pnts:
                lon, lat = cur[0].getPart().X, cur[0].getPart().Y
                np = GetNearPoint(tot_pnt_list, lon, lat)
                ptList.append((row[0], cur[1], np)) # PostCode ID, Point ID, X/Y
                del cur
            arcpy.Delete_management(PostCode_Polygon_CopyFeatures, "")
            arcpy.Delete_management(PostcodeWaterbodyUnion_CopyFeatures_Sig, "")
            arcpy.Delete_management(PostcodeWaterbodyUnion_CopyFeatures_Sig2, "")
            arcpy.Delete_management(PostcodeWaterbodyPoints_CopyFeatures, "")
            del row
        except Exception as e:
            print(e)
    # Generate point shapefile
    spatialRef = arcpy.Describe(LocationPoint).spatialReference
    arcpy.CreateFeatureclass_management(os.path.dirname(OutputPoints), os.path.basename(OutputPoints), "POINT")
    arcpy.AddField_management(OutputPoints, "POSTID", "LONG", "", "", "", "", "NULLABLE", "NON_REQUIRED", "")
    arcpy.AddField_management(OutputPoints, "PNTID", "LONG", "", "", "", "", "NULLABLE", "NON_REQUIRED", "")
    pcur = arcpy.da.InsertCursor(OutputPoints, ["SHAPE@", "POSTID", "PNTID"])
    array = arcpy.Array()
    for p in ptList:
        pcur.insertRow([p[2], p[0], p[1]])
    arcpy.DefineProjection_management(OutputPoints, spatialRef)
    arcpy.Delete_management(PostCodePolygon_Copy, "")
    arcpy.Delete_management(Postcode_Waterbody_Union, "")
    print("Done!")
else:
    print("Set all inputs/outputs as shapefile format")
t1 = time.time()
print(str(round((t1 - t0)/60,2))+" minutes")
