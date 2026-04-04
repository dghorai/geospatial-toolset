#! C:\Python27\ArcGISx6410.6\python.exe

# Remove Stair-Step Effect from DTM
# Author : Debabrata Ghorai
# Date : 27/11/2018

"""
Reference :
Kinsey-Henderson, A.E., Correcting height differences in digital elevation models derived from overlapping LiDAR survey runs, 2009. 
18th World IMACS/MODSIM Congress, Cairns, Australia 13-17 July 2009. Available at: http://mssanz.ogr.au/modsim09.
"""

import os
import time
import arcpy
import numpy
import shutil
import datetime

print ("%s" % time.ctime())
print ("Get Stair-Step Free DTM (SSFD)")
t0 = time.time()

# config env
arcpy.env.overwriteOutput = True
arcpy.CheckOutExtension("Spatial")
arcpy.CheckOutExtension("3D")

# Input arguments
intpType = "NNI" # use either NNI or TIN interpolation
gdb = r"D:\work\Output.gdb" # workspace
grids = r"D:\work\Inputs.gdb\Grid" # 50k x 50k grid with 100m buffer
contours = r"D:\work\Inputs.gdb\Gap1m" # 1m contour data
dtm = r"D:\work\Inputs.gdb\DTM" # original dtm/dem

# Get time notification
def timeit(ptxt, timesec):
    sec = datetime.timedelta(seconds=int(timesec))
    dty = datetime.datetime(1,1,1) + sec
    if (dty.day - 1) > 0:
        dtxt = ("%d days %d hours %d minutes %d seconds" % (dty.day-1, dty.hour, dty.minute, dty.second))
    elif dty.hour > 0:
        dtxt = ("%d hours %d minutes %d seconds" % (dty.hour, dty.minute, dty.second))
    elif dty.minute > 0:
        dtxt = ("%d minutes %d seconds" % (dty.minute, dty.second))
    else:
        dtxt = ("%d seconds" % dty.second)
    print (str(ptxt)+str(dtxt))
    return

# Temporary files
rast1 = arcpy.Raster(dtm)
select_grid = gdb+"\\"+"select_grid"
contours2 = gdb+"\\"+"contours"
dtm2rast = gdb+"\\"+"dtm_to_rast"
dtm2mask = gdb+"\\"+"dtm_to_mask"
bnd2mask = gdb+"\\"+"bnd_to_mask"
contour2rast = gdb+"\\"+"contour_to_rast"
rast2points = gdb+"\\"+"rast_to_points"
if os.path.exists(os.path.dirname(gdb)+"\\"+"DGFOLDER"):
    shutil.rmtree(os.path.dirname(gdb)+"\\"+"DGFOLDER")
os.mkdir(os.path.dirname(gdb)+"\\"+"DGFOLDER")
tin = os.path.dirname(gdb)+"\\"+"DGFOLDER"+"\\"+"TinCreate"
tin2rast = gdb+"\\"+"tin2rast"

# Loop over the grids
ssfdList = list()
rows = arcpy.SearchCursor(grids)
arcpy.MakeFeatureLayer_management(grids, "lyr")

for row in rows:
    t1 = time.time()
    if arcpy.Exists(select_grid):
        arcpy.Delete_management(select_grid, "")
    if arcpy.Exists(contours2):
        arcpy.Delete_management(contours2, "")
    if arcpy.Exists(dtm2rast):
        arcpy.Delete_management(dtm2rast, "")
    if arcpy.Exists(dtm2mask):
        arcpy.Delete_management(dtm2mask, "")
    if arcpy.Exists(bnd2mask):
        arcpy.Delete_management(bnd2mask, "")
    if arcpy.Exists(contour2rast):
        arcpy.Delete_management(contour2rast, "")
    if arcpy.Exists(rast2points):
        arcpy.Delete_management(rast2points, "")
    if arcpy.Exists(tin):
        arcpy.Delete_management(tin, "")
    if arcpy.Exists(tin2rast):
        arcpy.Delete_management(tin2rast, "")

    comid = row.COMID # mandatory field
    outputdtm = gdb+"\\"+"newDTM_"+str(comid)
    print("Grid Number : %s" % str(comid))
    
    print("\tCopy selected grid")
    arcpy.SelectLayerByAttribute_management("lyr", "NEW_SELECTION", "\"COMID\" = "+str(comid)+"")
    arcpy.CopyFeatures_management("lyr", select_grid)
    arcpy.SelectLayerByAttribute_management("lyr", "CLEAR_SELECTION")
    
    print("\tClip contours")
    arcpy.Clip_analysis(contours, select_grid, contours2, "0.001 Meters")
    desc1 = arcpy.Describe(contours2)
    sr = desc1.spatialReference
    
    print("\tSubset DEM")
    tempEnvironment0 = arcpy.env.snapRaster
    arcpy.env.snapRaster = dtm
    tempEnvironment1 = arcpy.env.extent
    arcpy.env.extent = desc1.Extent
    arcpy.gp.ExtractByMask_sa(dtm, select_grid, dtm2rast)
    arcpy.env.snapRaster = tempEnvironment0
    arcpy.env.extent = tempEnvironment1
    
    print("\tCreate DEM mask")
    arcpy.gp.IsNull_sa(dtm2rast, dtm2mask)
    arcpy.gp.Reclassify_sa(dtm2mask, "Value", "0 1;1 NODATA", bnd2mask, "DATA")
    
    print("\tConvert contours to raster")
    tempEnvironment0 = arcpy.env.snapRaster
    arcpy.env.snapRaster = dtm
    tempEnvironment1 = arcpy.env.extent
    arcpy.env.extent = desc1.Extent
    arcpy.gp.ExtractByMask_sa(dtm, contours2, contour2rast)
    arcpy.env.snapRaster = tempEnvironment0
    arcpy.env.extent = tempEnvironment1
    
    print("\tConvert raster to center points")
    arcpy.RasterToPoint_conversion(contour2rast, rast2points, "Value")
    arcpy.DefineProjection_management(rast2points, sr)

    if intpType == "NNI":
        print("\tNatural Neighbor Interpolation")
        tempEnvironment0 = arcpy.env.snapRaster
        arcpy.env.snapRaster = dtm
        tempEnvironment1 = arcpy.env.extent
        arcpy.env.extent = desc1.Extent
        arcpy.gp.NaturalNeighbor_sa(rast2points, "grid_code", tin2rast, str(rast1.meanCellWidth))
        arcpy.env.snapRaster = tempEnvironment0
        arcpy.env.extent = tempEnvironment1
    else:
        print("\tCreate TIN")
        tempEnvironment0 = arcpy.env.snapRaster
        arcpy.env.snapRaster = dtm
        tempEnvironment1 = arcpy.env.extent
        arcpy.env.extent = desc1.Extent
        arcpy.CreateTin_3d(tin, desc1.spatialReference, rast2points+" grid_code Mass_Points grid_code", "DELAUNAY")
        arcpy.env.snapRaster = tempEnvironment0
        arcpy.env.extent = tempEnvironment1
        
        print("\tConvert TIN to raster")
        tempEnvironment0 = arcpy.env.snapRaster
        arcpy.env.snapRaster = dtm
        tempEnvironment1 = arcpy.env.extent
        arcpy.env.extent = desc1.Extent
        arcpy.TinRaster_3d(tin, tin2rast, "FLOAT", "LINEAR", "CELLSIZE "+str(rast1.meanCellWidth), "1")
        arcpy.env.snapRaster = tempEnvironment0
        arcpy.env.extent = tempEnvironment1

    print("\tExport Raster")
    if arcpy.Exists(outputdtm):
        arcpy.Delete_management(outputdtm, "")
    tempEnvironment0 = arcpy.env.snapRaster
    arcpy.env.snapRaster = dtm
    tempEnvironment1 = arcpy.env.extent
    arcpy.env.extent = desc1.Extent
    arcpy.gp.ExtractByMask_sa(tin2rast, bnd2mask, outputdtm)
    arcpy.env.snapRaster = tempEnvironment0
    arcpy.env.extent = tempEnvironment1
    ssfdList.append(outputdtm)
    t2 = time.time() - t1
    timeit("\tProcess Time : ", t2)

if len(ssfdList) > 1:
    print("Mosaic Tiles")
    tempEnvironment0 = arcpy.env.snapRaster
    arcpy.env.snapRaster = dtm
    tempEnvironment1 = arcpy.env.extent
    arcpy.env.extent = dtm
    arcpy.MosaicToNewRaster_management(";".join(str(i) for i in ssfdList), gdb, intpType+"_Output", "", "32_BIT_FLOAT", "", "1", "BLEND", "FIRST")
    arcpy.env.snapRaster = tempEnvironment0
    arcpy.env.extent = tempEnvironment1
    for j in ssfdList:
        if arcpy.Exists(j):
            arcpy.Delete_management(j, "")
        del j
else:
    pass
    
print("Delete intermediate files")
shutil.rmtree(os.path.dirname(gdb)+"\\"+"DGFOLDER")
if arcpy.Exists(select_grid):
    arcpy.Delete_management(select_grid, "")
if arcpy.Exists(contours2):
    arcpy.Delete_management(contours2, "")
if arcpy.Exists(dtm2rast):
    arcpy.Delete_management(dtm2rast, "")
if arcpy.Exists(dtm2mask):
    arcpy.Delete_management(dtm2mask, "")
if arcpy.Exists(bnd2mask):
    arcpy.Delete_management(bnd2mask, "")
if arcpy.Exists(contour2rast):
    arcpy.Delete_management(contour2rast, "")
if arcpy.Exists(rast2points):
    arcpy.Delete_management(rast2points, "")
if arcpy.Exists(tin2rast):
    arcpy.Delete_management(tin2rast, "")
t3 = time.time() - t0
timeit("Total Process Time : ", t3)
