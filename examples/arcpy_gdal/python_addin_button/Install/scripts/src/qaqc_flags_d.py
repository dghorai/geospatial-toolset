# Import arcpy module
import arcpy

def XSCL_QAQC_Layer_D(inmfl, in_xscl, workspace):
    try:
        mfl_output = workspace+"\\"+"mfl_wo_xscl"
        inmfl_lyr = arcpy.MakeFeatureLayer_management(inmfl, "fl_Layer", "", "", "")

        if arcpy.Exists(mfl_output):
            arcpy.Delete_management(mfl_output, "")

        arcpy.AddMessage("Flowlines Without XSCL")
        arcpy.SelectLayerByLocation_management(inmfl_lyr, "INTERSECT", in_xscl, "0.001 Meters", "NEW_SELECTION")
        arcpy.SelectLayerByLocation_management(inmfl_lyr, "INTERSECT", "", "", "SWITCH_SELECTION")
        arcpy.CopyFeatures_management(inmfl_lyr, mfl_output, "", "0", "0", "0")
        arcpy.SelectLayerByAttribute_management(inmfl_lyr, "CLEAR_SELECTION")

        if arcpy.Exists(inmfl_lyr):
            arcpy.Delete_management(inmfl_lyr, "")

    except Exception as e:
        arcpy.AddMessage(e.message)
        
    return

if __name__ == '__main__':
    import sys
    XSCL_QAQC_Layer_D(sys.argv[1], sys.argv[2], sys.argv[3])
