# Import arcpy module
import arcpy

def XSCL_QAQC_Layer_A(inmfl, workspace):
    try:
        arcpy.AddMessage("Broken Model Flowlines")
        smtable = workspace+"\\"+"summaty_table"
        
        if arcpy.Exists(smtable):
            arcpy.Delete_management(smtable, "")
        
        arcpy.Statistics_analysis(inmfl, smtable, "UNIQUEID COUNT", "UNIQUEID")
        arcpy.JoinField_management(inmfl, "UNIQUEID", smtable, "UNIQUEID", "COUNT_UNIQUEID")
        arcpy.FeatureClassToFeatureClass_conversion(inmfl, workspace, "broken_mfl_output", "\"COUNT_UNIQUEID\" > 1", "", "")

        if arcpy.Exists(smtable):
            arcpy.Delete_management(smtable, "")

        fld2 = arcpy.ListFields(inmfl)
        for ii in fld2:
            if ii.name == "COUNT_UNIQUEID":
                arcpy.DeleteField_management(inmfl, ["COUNT_UNIQUEID"])
            else:
                pass
            del ii
        del fld2
            
    except Exception as e:
        arcpy.AddMessage(e.message)

    return

if __name__ == '__main__':
    import sys
    XSCL_QAQC_Layer_A(sys.argv[1], sys.argv[2])
