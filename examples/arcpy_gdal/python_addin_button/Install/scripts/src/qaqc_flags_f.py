# Import arcpy module
import arcpy
import mfl_filter
import polygon2line

def XSCL_QAQC_Layer_F(workspace, in_xscl, in_mfl, in_bnd, MXL, MXLH):
    try:
        model_output_3 = workspace+"\\"+"t_model1" ##
        out_mfl_3 = workspace+"\\"+"t_model2"
        xs_lyr = arcpy.MakeFeatureLayer_management(in_xscl, "xs_Layer", "", "", "")
        out_xscl_3 = workspace+"\\"+"t_xscl1"
        model_output_4 = workspace+"\\"+"t_xscl2" ##
        out_junc = workspace+"\\"+"mfl_junc"
        out_junc_buf = workspace+"\\"+"mfl_junc_buf"
        model_output_5 = workspace+"\\"+"t_model3" ##
        model_output_6 = workspace+"\\"+"t_model4" ##
        mfl_lyr = arcpy.MakeFeatureLayer_management(in_mfl, "mfl_Layer", "", "", "")
        model_output_7 = workspace+"\\"+"t_model5" ##
        out_bnd_line = workspace+"\\"+"out_bnd_ln"
        model_output_8 = workspace+"\\"+"t_model6" ##
        model_output_9_tmp = workspace+"\\"+"tmp_mfl_flag"
        model_output_9 = workspace+"\\"+"t_model7" ##

        if arcpy.Exists(model_output_3):
            arcpy.Delete_management(model_output_3, "")
        if arcpy.Exists(out_mfl_3):
            arcpy.Delete_management(out_mfl_3, "")
        if arcpy.Exists(out_xscl_3):
            arcpy.Delete_management(out_xscl_3, "")
        if arcpy.Exists(model_output_4):
            arcpy.Delete_management(model_output_4, "")
        if arcpy.Exists(out_junc):
            arcpy.Delete_management(out_junc, "")
        if arcpy.Exists(out_junc_buf):
            arcpy.Delete_management(out_junc_buf, "")
        if arcpy.Exists(model_output_5):
            arcpy.Delete_management(model_output_5, "")
        if arcpy.Exists(model_output_6):
            arcpy.Delete_management(model_output_6, "")
        if arcpy.Exists(model_output_7):
            arcpy.Delete_management(model_output_7, "")
        if arcpy.Exists(out_bnd_line):
            arcpy.Delete_management(out_bnd_line, "")
        if arcpy.Exists(model_output_8):
            arcpy.Delete_management(model_output_8, "")
        if arcpy.Exists(model_output_9_tmp):
            arcpy.Delete_management(model_output_9_tmp, "")
        if arcpy.Exists(model_output_9):
            arcpy.Delete_management(model_output_9, "")
        
        arcpy.AddMessage("XSCL Length is LT "+str(MXL)+" m of All River Order")
        fields = arcpy.ListFields(in_xscl)
        for f in fields:
            if f.name == "XLENGTH":
                pass
            else:
                arcpy.AddField_management(in_xscl, "XLENGTH", "DOUBLE", "", "", "", "", "NULLABLE", "REQUIRED")
        arcpy.CalculateField_management(in_xscl, "XLENGTH", "!shape.length!", "PYTHON_9.3")
        arcpy.FeatureClassToFeatureClass_conversion(in_xscl, workspace, "t_model1", "\"XLENGTH\" < "+str(MXL), "", "")
        # MXL = Minimum XSCL Length

        arcpy.AddMessage("XSCL Length is LT "+str(MXLH)+" m of River Order 3rd and Above")
        arcpy.FeatureClassToFeatureClass_conversion(in_mfl, workspace, "t_model2", "\"Order_\" >= 3", "", "")
        arcpy.SelectLayerByLocation_management(xs_lyr, "INTERSECT", out_mfl_3, "0.001 Meters", "NEW_SELECTION")
        arcpy.CopyFeatures_management(xs_lyr, out_xscl_3, "", "0", "0", "0")
        arcpy.SelectLayerByAttribute_management(xs_lyr, "CLEAR_SELECTION")
        arcpy.FeatureClassToFeatureClass_conversion(out_xscl_3, workspace, "t_xscl2", "\"XLENGTH\" < "+str(MXLH), "", "")
        # MXLH = Minimum XSCL Length at Higher Order
        
        arcpy.AddMessage("XSCL is Near to MFL Junction")
        arcpy.Intersect_analysis(str(in_mfl)+" #", out_junc, "ALL", "0.001 Meters", "POINT")
        arcpy.Buffer_analysis(out_junc, out_junc_buf, "100 Meters", "FULL", "ROUND", "NONE", "")
        arcpy.SelectLayerByLocation_management(xs_lyr, "INTERSECT", out_junc_buf, "0.001 Meters", "NEW_SELECTION")
        arcpy.CopyFeatures_management(xs_lyr, model_output_5, "", "0", "0", "0")
        arcpy.SelectLayerByAttribute_management(xs_lyr, "CLEAR_SELECTION")

        arcpy.AddMessage("XSCL is Out of Model Boundary")
        arcpy.SelectLayerByLocation_management(xs_lyr, "INTERSECT", in_bnd, "0.001 Meters", "NEW_SELECTION")
        arcpy.SelectLayerByLocation_management(xs_lyr, "INTERSECT", "", "", "SWITCH_SELECTION")
        arcpy.CopyFeatures_management(xs_lyr, model_output_6, "", "0", "0", "0")
        arcpy.SelectLayerByAttribute_management(xs_lyr, "CLEAR_SELECTION")

        arcpy.AddMessage("MFL is Out of Model Boundary")
        arcpy.SelectLayerByLocation_management(mfl_lyr, "INTERSECT", in_bnd, "0.001 Meters", "NEW_SELECTION")
        arcpy.SelectLayerByLocation_management(mfl_lyr, "INTERSECT", "", "", "SWITCH_SELECTION")
        arcpy.CopyFeatures_management(mfl_lyr, model_output_7, "", "0", "0", "0")
        arcpy.SelectLayerByAttribute_management(mfl_lyr, "CLEAR_SELECTION")

        arcpy.AddMessage("XSCL is Crosses the Model Boundary")
        polygon2line.Polygon_to_Line(in_bnd, out_bnd_line)
        arcpy.SelectLayerByLocation_management(xs_lyr, "INTERSECT", out_bnd_line, "0.001 Meters", "NEW_SELECTION")
        arcpy.CopyFeatures_management(xs_lyr, model_output_8, "", "0", "0", "0")
        arcpy.SelectLayerByAttribute_management(xs_lyr, "CLEAR_SELECTION")

        arcpy.AddMessage("MFL is Crosses the Model Boundary")
        arcpy.SelectLayerByLocation_management(mfl_lyr, "INTERSECT", out_bnd_line, "0.001 Meters", "NEW_SELECTION")
        arcpy.CopyFeatures_management(mfl_lyr, model_output_9_tmp, "", "0", "0", "0")
        arcpy.SelectLayerByAttribute_management(mfl_lyr, "CLEAR_SELECTION")
        mfl_filter.MFL_Crosses_Filter(model_output_9_tmp, in_bnd, model_output_9)

        if arcpy.Exists(out_mfl_3):
            arcpy.Delete_management(out_mfl_3, "")
        if arcpy.Exists(out_xscl_3):
            arcpy.Delete_management(out_xscl_3, "")
        if arcpy.Exists(out_junc):
            arcpy.Delete_management(out_junc, "")
        if arcpy.Exists(out_junc_buf):
            arcpy.Delete_management(out_junc_buf, "")
        if arcpy.Exists(out_bnd_line):
            arcpy.Delete_management(out_bnd_line, "")
        if arcpy.Exists(model_output_9_tmp):
            arcpy.Delete_management(model_output_9_tmp, "")
        if arcpy.Exists(xs_lyr):
            arcpy.Delete_management(xs_lyr, "")
        if arcpy.Exists(mfl_lyr):
            arcpy.Delete_management(mfl_lyr, "")

    except Exception as e:
        arcpy.AddMessage(e.message)

    return

if __name__ == '__main__':
    import sys
    XSCL_QAQC_Layer_F(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
