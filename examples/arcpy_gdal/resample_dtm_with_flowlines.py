# coding: utf-8

# Resample DEM with digitized river channel to keep original channel elevation on the resmapled DEM 
# Date: August 8, 2019

# Import Modules
import arcpy

# config env
arcpy.env.overwriteOutput = True
arcpy.CheckOutExtension("Spatial")

# Inputs
res_cell = 25
workspace = r"D:\DGhorai\Resample_Output.gdb"
in_org_dem = r"D:\DGhorai\Outputs.gdb\FinalDTM"
in_dgtz_flowline = r"D:\DGhorai\Outputs.gdb\Flowlines"
in_snap_dem = r"D:\DGhorai\Snap_Raster.gdb\SnapRast_DTM"


out_res_dem = workspace+"\\res_dem"
out_mfl_dem = workspace+"\\mfl_dem"
out_agg_mfl = workspace+"\\agg_mfl"
out_agg_pnt = workspace+"\\agg_pnt"
out_pnt_rst = workspace+"\\pnt_rst"

# Resample
tempEnvironment0 = arcpy.env.snapRaster
arcpy.env.snapRaster = in_snap_dem
arcpy.Resample_management(in_org_dem, out_res_dem, str(res_cell)+" "+str(res_cell), "BILINEAR")
arcpy.env.snapRaster = tempEnvironment0

# Get original cell values along river channel
tempEnvironment0 = arcpy.env.snapRaster
arcpy.env.snapRaster = in_org_dem
arcpy.gp.ExtractByMask_sa(in_org_dem, in_dgtz_flowline, out_mfl_dem)
arcpy.env.snapRaster = tempEnvironment0

# Aggregate MFL cell values
description = arcpy.Describe(in_org_dem)
org_cellsize = description.children[0].meanCellHeight
cell_factor = int(res_cell/org_cellsize)
tempEnvironment0 = arcpy.env.snapRaster
arcpy.env.snapRaster = out_res_dem
arcpy.gp.Aggregate_sa(out_mfl_dem, out_agg_mfl, str(cell_factor), "MINIMUM", "TRUNCATE", "DATA")
arcpy.env.snapRaster = tempEnvironment0

# Convert MFL aggregate cell value to point
tempEnvironment0 = arcpy.env.snapRaster
arcpy.env.snapRaster = out_res_dem
arcpy.RasterToPoint_conversion(out_agg_mfl, out_agg_pnt, "Value")
arcpy.env.snapRaster = tempEnvironment0

# Convert point to raster
tempEnvironment0 = arcpy.env.snapRaster
arcpy.env.snapRaster = out_res_dem
arcpy.PointToRaster_conversion(out_agg_pnt, "GRID_CODE", out_pnt_rst, "MOST_FREQUENT", "NONE", str(res_cell))
arcpy.env.snapRaster = tempEnvironment0

# Replace new cell value to resample DEM
tempEnvironment0 = arcpy.env.snapRaster
arcpy.env.snapRaster = out_res_dem
arcpy.MosaicToNewRaster_management(str(out_res_dem)+";"+str(out_pnt_rst), workspace, "Resample_DEM", "", "32_BIT_FLOAT", "", "1", "LAST", "FIRST")
arcpy.env.snapRaster = tempEnvironment0

# Delete files
arcpy.Delete_management(out_res_dem, "")
arcpy.Delete_management(out_mfl_dem, "")
arcpy.Delete_management(out_agg_mfl, "")
arcpy.Delete_management(out_agg_pnt, "")
