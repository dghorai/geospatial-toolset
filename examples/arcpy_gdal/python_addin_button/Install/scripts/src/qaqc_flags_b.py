# Import arcpy module
import arcpy

def XSCL_QAQC_Layer_B(inmfl, workspace, MRL):
    try:
        arcpy.AddMessage("Model Flowlines LT "+str(MRL)+" m Length")
        fields = arcpy.ListFields(inmfl)
        for f in fields:
            if f.name == "RLENGTH":
                pass
            else:
                arcpy.AddField_management(inmfl, "RLENGTH", "DOUBLE", "", "", "", "", "NULLABLE", "REQUIRED")
            del f
        arcpy.CalculateField_management(inmfl, "RLENGTH", "!shape.length!", "PYTHON_9.3")
        arcpy.FeatureClassToFeatureClass_conversion(inmfl, workspace, "mrl_mfl_output", "\"RLENGTH\" < "+str(MRL), "", "")
        
    except Exception as e:
        arcpy.AddMessage(e.message)

    return

if __name__ == '__main__':
    import sys
    XSCL_QAQC_Layer_B(sys.argv[1], sys.argv[2], sys.argv[3])
