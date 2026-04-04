# Import arcpy module
import arcpy

def XSCL_QAQC_Layer_H(workspace, in_xscl, in_dtm):
    try:
        temp_out1 = workspace+"\\"+"xs_to_endpnt1"
        temp_out2 = workspace+"\\"+"xs_to_endpnt2"
        temp_out3_str = workspace+"\\"+"temp_out3_str11"
        xs_lyr = arcpy.MakeFeatureLayer_management(in_xscl, "xs_Layer", "", "", "")
        model_output_16 = workspace+"\\"+"t_model16" ##

        if arcpy.Exists(temp_out1):
            arcpy.Delete_management(temp_out1, "")
        if arcpy.Exists(temp_out2):
            arcpy.Delete_management(temp_out2, "")
        if arcpy.Exists(temp_out3_str):
            arcpy.Delete_management(temp_out3_str, "")
        if arcpy.Exists(model_output_16):
            arcpy.Delete_management(model_output_16, "")
        
        arcpy.AddMessage("XSCL Passing Through NoData Elevation")
        sr = arcpy.Describe(in_xscl).spatialReference
        array = arcpy.da.FeatureClassToNumPyArray(in_xscl, ["SHAPE@XY"], spatial_reference = sr, explode_to_points = True)
        if array.size == 0:
            arcpy.AddError(in_xscl + "has no features.")
        else:
            arcpy.da.NumPyArrayToFeatureClass(array, temp_out1, ['SHAPE@XY'], sr)
            arcpy.gp.ExtractValuesToPoints_sa(temp_out1, in_dtm, temp_out2, "NONE", "VALUE_ONLY")
            arcpy.FeatureClassToFeatureClass_conversion(temp_out2, workspace, "temp_out3_str11", "\"RASTERVALU\" =  -9999", "", "")
            arcpy.SelectLayerByLocation_management(xs_lyr, "INTERSECT", temp_out3_str, "0.001 Meters", "NEW_SELECTION")
            arcpy.CopyFeatures_management(xs_lyr, model_output_16, "", "0", "0", "0")
            arcpy.SelectLayerByAttribute_management(xs_lyr, "CLEAR_SELECTION")

        if arcpy.Exists(temp_out1):
            arcpy.Delete_management(temp_out1, "")
        if arcpy.Exists(temp_out2):
            arcpy.Delete_management(temp_out2, "")
        if arcpy.Exists(temp_out3_str):
            arcpy.Delete_management(temp_out3_str, "")
        if arcpy.Exists(xs_lyr):
            arcpy.Delete_management(xs_lyr, "")

    except Exception as e:
        arcpy.AddMessage(e.message)

    return

if __name__ == '__main__':
    import sys
    XSCL_QAQC_Layer_H(sys.argv[1], sys.argv[2], sys.argv[3])
