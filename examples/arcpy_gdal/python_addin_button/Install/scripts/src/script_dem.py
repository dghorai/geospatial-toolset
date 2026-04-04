# Import arcpy module
import arcpy
arcpy.env.overwriteOutput = True

def XSCL_QAQC_Layer_I(workspace, in_xscl, in_dtm, model_output_17):
    try:
        xs_lyr = arcpy.MakeFeatureLayer_management(in_xscl, "xs_Layer", "", "", "")
        #model_output_17 = workspace+"\\"+"output_min_dem"

        if arcpy.Exists(model_output_17):
            arcpy.Delete_management(model_output_17, "")
        
        # Line to end point
        arcpy.AddMessage("Minimum Value at the End of XSCL")
        sr = arcpy.Describe(in_xscl).spatialReference
        segPts = arcpy.CreateFeatureclass_management(workspace, "xs2point", 'POINT','','','',sr)
        arcpy.AddField_management(workspace+"\\"+"xs2point", "XSCL_UID", "TEXT", "", "", 25)
        icursor = arcpy.da.InsertCursor(segPts, ('SHAPE@', 'XSCL_UID'))
        rows = arcpy.da.SearchCursor(in_xscl, ('SHAPE@', 'XSCL_UID'))
        for row in rows:
            plist = list()
            for part in row[0]:
                for pnt in part:
                    if pnt:
                        plist.append((pnt.X, pnt.Y))
                    else:
                        print ("Interior Ring")
                    del pnt
                del part
            nd = (plist[0], plist[-1])
            for i in nd:
                icursor.insertRow((i, row[1]))
                del i
            del nd
            del row
        del rows
        del icursor

        # Get DTM cell values at points
        arcpy.AddMessage("Extract Values to Points")
        arcpy.gp.ExtractValuesToPoints_sa(workspace+"\\"+"xs2point", in_dtm, workspace+"\\"+"xs2point2", "NONE", "VALUE_ONLY")

        # Process: Zonal Statistics as Table
        tempEnvironment0 = arcpy.env.snapRaster
        arcpy.env.snapRaster = in_dtm
        tempEnvironment1 = arcpy.env.extent
        arcpy.env.extent = arcpy.Describe(in_xscl).extent
        arcpy.gp.ZonalStatisticsAsTable_sa(in_xscl, "XSCL_UID", in_dtm, workspace+"\\"+"zone_txt", "DATA", "MINIMUM")
        arcpy.env.snapRaster = tempEnvironment0
        arcpy.env.extent = tempEnvironment1

        # Compare values
        arcpy.AddMessage("Join Field")
        arcpy.JoinField_management(workspace+"\\"+"xs2point2", "XSCL_UID", workspace+"\\"+"zone_txt", "XSCL_UID", "MIN")

        # Get the flag xscl
        fields = arcpy.ListFields(workspace+"\\"+"xs2point2")
        for f in fields:
            if f.name == "RMATCH":
                pass
            else:
                arcpy.AddField_management(workspace+"\\"+"xs2point2", "RMATCH", "DOUBLE", "", "", "", "", "NULLABLE", "REQUIRED")
        arcpy.CalculateField_management(workspace+"\\"+"xs2point2", "RMATCH", "!RASTERVALU! - !MIN!", "PYTHON_9.3")

        # Query - if RMATCH == 0 then flag as xscl end node has same as min value
        arcpy.AddMessage("Select Records")
        xslyr = arcpy.MakeFeatureLayer_management(workspace+"\\"+"xs2point2", 'xslyr2', "", "", "")
        arcpy.SelectLayerByAttribute_management(xslyr, "NEW_SELECTION", "\"RMATCH\" =0")
        arcpy.CopyFeatures_management(xslyr, workspace+"\\"+"xs_point3", "", "0", "0", "0")
        arcpy.SelectLayerByAttribute_management(xslyr, "CLEAR_SELECTION")
        arcpy.SelectLayerByLocation_management(xs_lyr, "INTERSECT", workspace+"\\"+"xs_point3", "0.001 Meters", "NEW_SELECTION")
        arcpy.CopyFeatures_management(xs_lyr, model_output_17, "", "0", "0", "0")
        arcpy.SelectLayerByAttribute_management(xs_lyr, "CLEAR_SELECTION")

        arcpy.Delete_management(xslyr, "")

        if arcpy.Exists(workspace+"\\"+"xs2point"):
            arcpy.Delete_management(workspace+"\\"+"xs2point", "")
        if arcpy.Exists(workspace+"\\"+"zone_txt"):
            arcpy.Delete_management(workspace+"\\"+"zone_txt", "")
        if arcpy.Exists(workspace+"\\"+"xs2point2"):
            arcpy.Delete_management(workspace+"\\"+"xs2point2", "")
        if arcpy.Exists(workspace+"\\"+"xs_point3"):
            arcpy.Delete_management(workspace+"\\"+"xs_point3", "")

        # Update field
        arcpy.AddMessage("Update Field")
        fld10 = arcpy.ListFields(model_output_17)
        for f in fld10:
            if f.name == "Comments":
                arcpy.CalculateField_management(model_output_17, "Comments", "\"Minimum Value at End\"", "PYTHON_9.3", "")
                pass
            else:
                arcpy.AddField_management(model_output_17, "Comments", "TEXT", "", "", 255)
                arcpy.CalculateField_management(model_output_17, "Comments", "\"Minimum Value at End\"", "PYTHON_9.3", "")
            del f
        del fld10

        # Delete field from input file
        fld12 = arcpy.ListFields(in_xscl)
        for ii in fld12:
            if ii.name == "MIN":
                arcpy.DeleteField_management(in_xscl, ["MIN"])
            elif ii.name == "RMATCH":
                arcpy.DeleteField_management(in_xscl, ["RMATCH"])
            else:
                pass
            del ii
        del fld12
        
        # Delete field from output file
        fld13 = arcpy.ListFields(model_output_17)
        for ii in fld13:
            if ii.name == "Comments":
                pass
            elif ii.name == "Shape":
                pass
            elif (ii.name == "OBJECTID") or (ii.name == "OBJECTID_1") or (ii.name == "OBJECTID_2"):
                pass
            elif (ii.name == "Shape_Length") or (ii.name == "Shape_Leng"):
                pass
            else:
                arcpy.DeleteField_management(model_output_17, [str(ii.name)])
            del ii
        del fld13

        if arcpy.Exists(xs_lyr):
            arcpy.Delete_management(xs_lyr, "")

    except Exception as e:
        arcpy.AddMessage(e.message)

    return

if __name__ == '__main__':
    import sys
    XSCL_QAQC_Layer_I(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
