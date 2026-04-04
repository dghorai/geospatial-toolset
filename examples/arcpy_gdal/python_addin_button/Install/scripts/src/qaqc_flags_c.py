# Import arcpy module
import arcpy

def XSCL_QAQC_Layer_C(inxscl, inmfl, workspace):
    try:
        arcpy.AddMessage("XSCL Without MFL")
        xscl_output = workspace+"\\"+"xscl_wo_mfl" ##
        xscl_lyr = arcpy.MakeFeatureLayer_management(inxscl, "xs_Layer", "", "", "")

        if arcpy.Exists(xscl_output):
            arcpy.Delete_management(xscl_output, "")
        
        arcpy.SelectLayerByLocation_management(xscl_lyr, "INTERSECT", inmfl, "0.001 Meters", "NEW_SELECTION")
        arcpy.SelectLayerByLocation_management(xscl_lyr, "INTERSECT", "", "", "SWITCH_SELECTION")
        arcpy.CopyFeatures_management(xscl_lyr, xscl_output, "", "0", "0", "0")
        arcpy.SelectLayerByAttribute_management(xscl_lyr, "CLEAR_SELECTION")

        if arcpy.Exists(xscl_lyr):
            arcpy.Delete_management(xscl_lyr, "")
        
    except Exception as e:
        arcpy.AddMessage(e.message)
        
    return

if __name__ == '__main__':
    import sys
    XSCL_QAQC_Layer_C(sys.argv[1], sys.argv[2], sys.argv[3])
    
