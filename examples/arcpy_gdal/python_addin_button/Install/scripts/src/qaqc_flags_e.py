# Import arcpy module
import arcpy

def XSCL_QAQC_Layer_E(in_xscl, in_mfl, workspace, wider_dist, close_dist):
    try:
        output_points = workspace+"\\"+"t_out_pnts"
        output_mfl_split = workspace+"\\"+"t_pnt_split"
        output_gt_rvl = workspace+"\\"+"t_wider_mfl" ##
        output_lt_rvl = workspace+"\\"+"t_closer_mfl" ##
        output_x = workspace+"\\"+"t_pnt_x"
        xslyr = arcpy.MakeFeatureLayer_management(in_xscl, "xs_Layer", "", "", "")
        output_xs = workspace+"\\"+"t_output_xs" ##
        out_stats = workspace+"\\summarize_stats"

        if arcpy.Exists(output_points):
            arcpy.Delete_management(output_points, "")
        if arcpy.Exists(output_mfl_split):
            arcpy.Delete_management(output_mfl_split, "")
        if arcpy.Exists(output_x):
            arcpy.Delete_management(output_x, "")
        if arcpy.Exists(output_gt_rvl):
            arcpy.Delete_management(output_gt_rvl, "")
        if arcpy.Exists(output_lt_rvl):
            arcpy.Delete_management(output_lt_rvl, "")
        if arcpy.Exists(output_xs):
            arcpy.Delete_management(output_xs, "")
        if arcpy.Exists(out_stats):
            arcpy.Delete_management(out_stats, "")
            
        fields = arcpy.ListFields(in_xscl)
        for f in fields:
            if f.name == "IMP":
                pass
            else:
                arcpy.AddField_management(in_xscl, "IMP", "LONG", "", "", "", "", "NULLABLE", "NON_REQUIRED", "")
        arcpy.CalculateField_management(in_xscl, "IMP", "!OBJECTID!", "PYTHON_9.3", "")
        arcpy.Intersect_analysis(in_xscl+" #;"+in_mfl+" #", output_points, "ALL", "0.001 Meters", "POINT")

        arcpy.AddMessage("No XSCL Within "+str(wider_dist)+"m Distance")
        arcpy.SplitLineAtPoint_management(in_mfl, output_points, output_mfl_split, "0.001 Meters")
        arcpy.FeatureClassToFeatureClass_conversion(output_mfl_split, workspace, "t_wider_mfl", "Shape_length > "+str(wider_dist), "", "")
        arcpy.AddMessage("Distance "+str(close_dist)+" m Between Two XSCL")
        arcpy.FeatureClassToFeatureClass_conversion(output_mfl_split, workspace, "t_closer_mfl", "Shape_length < "+str(close_dist), "", "")
        
        arcpy.AddMessage("Multi-intersect Between XSCL and MFL")
        arcpy.Statistics_analysis(output_points, out_stats, "IMP FIRST", "IMP")
        arcpy.JoinField_management(in_xscl, "IMP", out_stats, "IMP", "FREQUENCY")
        #arcpy.CalculateField_management(output_points, "IMP", "!shape.isMultipart!", "PYTHON_9.3")
        arcpy.FeatureClassToFeatureClass_conversion(in_xscl, workspace, "t_output_xs", "FREQUENCY > 1", "", "")
        #arcpy.SelectLayerByLocation_management(xslyr, "INTERSECT", output_x, "0.001 Meters", "NEW_SELECTION")
        #arcpy.CopyFeatures_management(xslyr, output_xs, "", "0", "0", "0")
        #arcpy.SelectLayerByAttribute_management(xslyr, "CLEAR_SELECTION")

        if arcpy.Exists(output_points):
            arcpy.Delete_management(output_points, "")
        if arcpy.Exists(output_mfl_split):
            arcpy.Delete_management(output_mfl_split, "")
        if arcpy.Exists(output_x):
            arcpy.Delete_management(output_x, "")
        if arcpy.Exists(xslyr):
            arcpy.Delete_management(xslyr, "")
        if arcpy.Exists(out_stats):
            arcpy.Delete_management(out_stats, "")

    except Exception as e:
        arcpy.AddMessage(e.message)
        
    return

if __name__ == '__main__':
    import sys
    XSCL_QAQC_Layer_E(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
