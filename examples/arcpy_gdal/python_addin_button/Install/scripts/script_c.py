# XSCL QAQC Toolbox
# =====================
"""
Identify/Flag XSCL which is one-sided short by 20% of the total length.
Date: 2016-07-15
Version: 1.0
Platform: Python 2.7, ArcPy
"""
# Import modules
import arcpy
arcpy.env.overwriteOutput = True

# Input arguments
#===================
Workspace = arcpy.GetParameterAsText(0)
inMfl = arcpy.GetParameterAsText(1)
inXscl = arcpy.GetParameterAsText(2)
sRATIO = float(arcpy.GetParameterAsText(3)) #25 # If the xscl bank length is LTE 25% then flag the XSCL
modelOutput = arcpy.GetParameterAsText(4)

# Function for XSCL QAQC Laye L
def XSCL_QAQC_Layer_L(workspace, in_xscl, in_mfl, RATIO, out2):
    try:
        print ("Cross-Section is One Side Short")
        arcpy.AddMessage("Cross-Section is One Side Short")
        
        # Temporary files
        out_point = workspace+"\\"+"out_points"
        out_xscl = workspace+"\\"+"out_xscls"
        inLineName = arcpy.Describe(in_xscl).name
        out1 = workspace+"\\"+inLineName+"_pts"

        # Delete files (if exists)
        if arcpy.Exists(out_point):
            arcpy.Delete_management(out_point, "")
        if arcpy.Exists(out_xscl):
            arcpy.Delete_management(out_xscl, "")
        if arcpy.Exists(out1):
            arcpy.Delete_management(out1, "")
        if arcpy.Exists(out2):
            arcpy.Delete_management(out2, "")

        # Add field
        fields = arcpy.ListFields(in_xscl)
        for f in fields:
            if f.name == "XS_LENGTH":
                pass
            else:
                arcpy.AddField_management(in_xscl, "XS_LENGTH", "DOUBLE", "", "", "", "", "NULLABLE", "REQUIRED")

        # Calculate field
        arcpy.CalculateField_management(in_xscl, "XS_LENGTH", "!shape.length!", "PYTHON_9.3") ####

        # Process: Intersect
        arcpy.AddMessage("Intersect Analysis") ##
        arcpy.Intersect_analysis(in_xscl+" #;"+in_mfl+" #", out_point, "ALL", "0.001 Meters", "POINT")

        # Process: Split
        arcpy.AddMessage("Split Line") ##
        arcpy.SplitLineAtPoint_management(in_xscl,out_point,out_xscl,"")

        # Add field
        fields2 = arcpy.ListFields(out_xscl)
        for f in fields2:
            if f.name == "BANK_LENGTH":
                pass
            else:
                arcpy.AddField_management(out_xscl, "BANK_LENGTH", "DOUBLE", "", "", "", "", "NULLABLE", "REQUIRED")
            del f
        del fields2

        # Calculate field
        arcpy.CalculateField_management(out_xscl, "BANK_LENGTH", "!shape.length!", "PYTHON_9.3")

        # Short XSCL Flag
        arcpy.AddMessage("Cross-Section Flag")
        flagList = list()
        rows = arcpy.da.SearchCursor(out_xscl, ("SHAPE@", "XS_LENGTH", "BANK_LENGTH"))
        for row in rows:
            xsLength = row[1]
            xsBankLength = row[2]
            p = int((xsBankLength/xsLength)*100)
            if p <= RATIO :
                t = list()
                for pnt in row[0].getPart()[0]:
                    t.append((pnt.X, pnt.Y))
                cnt = ((t[0][0]+t[-1][0])/2.0, (t[0][1]+t[-1][1])/2.0)
                flagList.append(cnt)
            del row
        del rows

        # Draw flag points
        sr = arcpy.Describe(in_xscl).spatialReference
        segPts = arcpy.CreateFeatureclass_management(workspace, inLineName+"_pts", 'POINT','','','',sr)
        icursor = arcpy.da.InsertCursor(segPts, ("SHAPE@"))
        for row in flagList:
            icursor.insertRow((row,))
            del row
        del icursor
        del flagList

        # Select xscl
        arcpy.AddMessage("Cross-Section Selection")
        try:
            xs_lyr = arcpy.MakeFeatureLayer_management(in_xscl, "xs1_Layer", "", "", "")
            arcpy.SelectLayerByLocation_management(xs_lyr, "INTERSECT", out1, "0.001 Meters", "NEW_SELECTION")
            arcpy.CopyFeatures_management(xs_lyr, out2, "", "0", "0", "0")
            arcpy.Delete_management(xs_lyr, "")
        except Exception as e:
            print (e.message)
            
        arcpy.Delete_management(out_point, "")
        arcpy.Delete_management(out_xscl, "")
        arcpy.Delete_management(out1, "")

        # Update field
        arcpy.AddMessage("Update Field")
        fld = arcpy.ListFields(out2)
        for f in fld:
            if f.name == "Comments":
                arcpy.CalculateField_management(out2, "Comments", "\"Cross-section is one side short\"", "PYTHON_9.3", "")
                pass
            else:
                arcpy.AddField_management(out2, "Comments", "TEXT", "", "", 255)
                arcpy.CalculateField_management(out2, "Comments", "\"Cross-section is one side short\"", "PYTHON_9.3", "")
            del f
        del fld

        # Delete field from input file
        fld12 = arcpy.ListFields(in_xscl)
        for ii in fld12:
            if ii.name == "XS_LENGTH":
                arcpy.DeleteField_management(in_xscl, ["XS_LENGTH"])
            else:
                pass
            del ii
        del fld12

        # Delete field from output file
        fld13 = arcpy.ListFields(out2)
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
                arcpy.DeleteField_management(out2, [str(ii.name)])
            del ii
        del fld13
                
    except Exception as e:
        print (e.message)
        arcpy.AddMessage(e.message)
        
    return

# Input/output files
if __name__ == '__main__':
    XSCL_QAQC_Layer_L(Workspace, inXscl, inMfl, sRATIO, modelOutput)
