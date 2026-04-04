#! C:\Python27\ArcGISx6410.6\python.exe

"""
Estimating Pixel-to-Pixel Travel Time using Particle Tracking Approach.
Scripted by: Debabrata Ghorai
Date: 2015-10-29
Python: 2.7
"""

# Import modules
import time
import arcpy
import numpy

# config env
arcpy.env.overwriteOutput = True
arcpy.CheckOutExtension("Spatial")

# Start time
print("Start : %s" % time.ctime())
start = time.time()


# PHASE-1
# Input Arguments
# ---------------
# Set "Cell Size" for the Travel Time Computation
cellSize = 90
# Set the path for Output/Working FGDB
gdb = r"D:\Projects_Working\Test.gdb"
# Set the path for Region Catchment Boundary
in_bnd = r"D:\Projects_Working\Test.gdb\Catchment"


# PHASE-2
# Set the path for FILL-DEM of the Region
in_dem = r"\\Models\Region01.gdb\DTM_90M_Fill"
# Set the path for FDR of the Region
in_fdr = r"\\Models\Region01.gdb\DTM_90M_Fill_FDR"
# Set the path for FACC of the Region
in_facc = r"\\Models\Region01.gdb\DTM_90M_Fill_FDR_FACC"
in_DelT = r"D:\Projects_Working\Test.gdb\dtm"

# Temporary files
select_bnd = gdb+"/"+"select_bnd_"
fill_dem = gdb+"/"+"fill_dem_"
fdr_rast = gdb+"/"+"fdr_rast_"
focaldem = gdb+"/"+"focal_dem_"
delvrast = gdb+"/"+"elev_rast_"
facc_rast = gdb+"/"+"facc_rast_"
travel_time_rast = gdb+"/"+"ttt_rast_"
travel_time_temp = gdb+"/"+"ttt_rast_temp"
catch_rast = gdb+"/"+"catch_rast"

# Geo-processing Started
print("Geo-processing Started")

arcpy.MakeFeatureLayer_management(in_bnd, "lyr")
rows = arcpy.SearchCursor(in_bnd)

for row in rows:
    startt = time.time()
    uniqueid = row.uniqueid
    print("uniqueid : %s" % str(uniqueid))
    arcpy.Delete_management("in_memory")
    print("\tDelete Temporary Files if Exists")
    if arcpy.Exists(select_bnd):
        arcpy.Delete_management(select_bnd, "")
    if arcpy.Exists(fill_dem):
        arcpy.Delete_management(fill_dem, "")
    if arcpy.Exists(fdr_rast):
        arcpy.Delete_management(fdr_rast, "")
    if arcpy.Exists(focaldem):
        arcpy.Delete_management(focaldem, "")
    if arcpy.Exists(delvrast):
        arcpy.Delete_management(delvrast, "")
    if arcpy.Exists(facc_rast):
        arcpy.Delete_management(facc_rast, "")
    if arcpy.Exists(travel_time_rast):
        arcpy.Delete_management(travel_time_rast, "")
    if arcpy.Exists(travel_time_temp):
        arcpy.Delete_management(travel_time_temp, "")

    output_travel_time = gdb+"\\"+"travel_time_"+str(uniqueid)
    arcpy.SelectLayerByAttribute_management("lyr", "NEW_SELECTION", "\"uniqueid\" = "+str(uniqueid)+"")
    arcpy.CopyFeatures_management("lyr", select_bnd)
    arcpy.SelectLayerByAttribute_management("lyr", "CLEAR_SELECTION")
    
    # Extract FDR by Mask
    print("\tClip FDR")
    tempEnvironment0 = arcpy.env.snapRaster
    arcpy.env.snapRaster = in_dem
    tempEnvironment1 = arcpy.env.extent
    arcpy.env.extent = arcpy.Describe(select_bnd).Extent
    arcpy.gp.ExtractByMask_sa(in_fdr, select_bnd, fdr_rast)
    arcpy.env.snapRaster = tempEnvironment0
    arcpy.env.extent = tempEnvironment1

    # Extract FACC by Mask
    print("\tClip FACC")
    tempEnvironment0 = arcpy.env.snapRaster
    arcpy.env.snapRaster = in_dem
    tempEnvironment1 = arcpy.env.extent
    arcpy.env.extent = arcpy.Describe(select_bnd).Extent
    arcpy.gp.ExtractByMask_sa(in_facc, select_bnd, facc_rast)
    arcpy.env.snapRaster = tempEnvironment0
    arcpy.env.extent = tempEnvironment1

    # Extract DeltaT by Mask
    print("\tClip DeltaT")
    tempEnvironment0 = arcpy.env.snapRaster
    arcpy.env.snapRaster = in_dem
    tempEnvironment1 = arcpy.env.extent
    arcpy.env.extent = arcpy.Describe(select_bnd).Extent
    arcpy.gp.ExtractByMask_sa(in_DelT, select_bnd, travel_time_temp)
    arcpy.env.snapRaster = tempEnvironment0
    arcpy.env.extent = tempEnvironment1
    # Extract DeltaT by Mask
    print("\tClip Dem")
    tempEnvironment0 = arcpy.env.snapRaster
    arcpy.env.snapRaster = in_dem
    tempEnvironment1 = arcpy.env.extent
    arcpy.env.extent = arcpy.Describe(select_bnd).Extent
    arcpy.gp.ExtractByMask_sa(in_dem, select_bnd, fill_dem)
    arcpy.env.snapRaster = tempEnvironment0
    arcpy.env.extent = tempEnvironment1
    #arcpy.PolygonToRaster_conversion (select_bnd, "uniqueid",catch_rast,cellsize=30)

    # Step-5: Get Pixel-to-Outlet Travel Time (travel_time)
    print("\tGet Pixel-to-Outlet Travel Time")
    fdr = arcpy.RasterToNumPyArray(fdr_rast)
    fac = arcpy.RasterToNumPyArray(facc_rast, nodata_to_value=-1)
    dtime = arcpy.RasterToNumPyArray(travel_time_temp, nodata_to_value=0)
    dem= arcpy.RasterToNumPyArray(fill_dem)
    
    print("\tAppend Pixel-to-Pixel Connection Index")
    dtime2=dtime.flatten('F')
    fac2=fac.flatten('F')
    fdr=fdr.flatten('F')
    inx=(-fac2).argsort()
    for inxx in inx:
        print(inxx)
        inx1=inxx+1
        if inx1%dtime.shape[0]==0:
            i=dtime.shape[0]-1
            j=(inx1/dtime.shape[0])-1
        else:
            i=inx1%dtime.shape[0]-1
            j=(inx1/dtime.shape[0])

        p=-1
        k=-1
        cd=fdr[inxx]
        
        if fac2[inxx]==-1:
            pass
        
        elif cd == 1:
            p=i
            k=j+1
        elif cd == 2:
            p=i+1
            k=j+1
        elif cd == 4:
            p=i+1
            k=j
        elif cd == 8:
            p=i+1
            k=j-1
            
        elif cd == 16:
            p=i
            k=j-1
        elif cd == 32:
            p=i-1
            k=j-1
        elif cd == 64:
            p=i-1
            k=j
        elif cd == 128:
            p=i-1
            k=j+1

        if inxx==inx[0] :
            print("C: ", inxx)
            dtime2[inxx]=0
        else:
            inx2=(p+1+((k)*dtime.shape[0]))-1
            print("D: ", inx2)
            dtime2[inxx]=dtime2[inxx]+dtime2[inx2]
            
    dtime2=numpy.reshape(dtime2,(dtime.shape[0],dtime.shape[1]),order="F")
    ras = arcpy.Raster(fdr_rast)
    raster_desc = arcpy.Describe(fdr_rast)
    raster_extent = raster_desc.Extent
    x_min = raster_extent.XMin
    y_min = raster_extent.YMin
    delta_time_cum = arcpy.NumPyArrayToRaster(dtime2, arcpy.Point(x_min, y_min), ras.meanCellWidth)
    delta_time_cum.save(travel_time_rast)
    arcpy.env.snapRaster = in_dem
    tempEnvironment1 = arcpy.env.extent
    arcpy.env.extent = arcpy.Describe(select_bnd).Extent
    arcpy.gp.ExtractByMask_sa(travel_time_rast, select_bnd, output_travel_time)
    arcpy.env.snapRaster = tempEnvironment0
    arcpy.env.extent = tempEnvironment1
    endt = time.time()
    print("\t", round((endt-startt)/60,2) , "minutes")

print("Delete Temporary Files")
if arcpy.Exists(select_bnd):
    arcpy.Delete_management(select_bnd, "")
if arcpy.Exists(fill_dem):
    arcpy.Delete_management(fill_dem, "")
if arcpy.Exists(fdr_rast):
    arcpy.Delete_management(fdr_rast, "")
if arcpy.Exists(focaldem):
    arcpy.Delete_management(focaldem, "")
if arcpy.Exists(delvrast):
    arcpy.Delete_management(delvrast, "")
if arcpy.Exists(facc_rast):
    arcpy.Delete_management(facc_rast, "")
if arcpy.Exists(travel_time_rast):
    arcpy.Delete_management(travel_time_rast, "")
        
 #End time
end = time.time()
print(round((end-start)/60,2) , "minutes")
print("End : %s" % time.ctime())
