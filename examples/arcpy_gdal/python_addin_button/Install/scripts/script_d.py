# Import arcpy module
import os
import re
import arcpy

arcpy.env.overwriteOutput = True

Workspace = arcpy.GetParameterAsText(0)
inCrosssection = arcpy.GetParameterAsText(1)
UXL = arcpy.GetParameterAsText(2)
qaqcOutput = arcpy.GetParameterAsText(3)

def XSCL_QAQC_Layer_J(workspace, in_xscl, T, model_output_18_str):
    try:
        print ("Begining From-Node XSCL Lemgth is LT 500 meters")
        arcpy.AddMessage("Begining From-Node XSCL Lemgth is LT "+str(T)+" Meters")

        folder = os.path.dirname(workspace)
        temp_dbf = folder+"\\"+"tempdbf.txt"
        j_table = workspace+"\\"+"j_table"
        if os.path.exists(temp_dbf):
            os.remove(temp_dbf)
            temp_dbf = folder+"\\"+"tempdbf.txt"

        # Loop over the xscl file
        rows = arcpy.da.SearchCursor(in_xscl, ("OBJECTID", "XSCL_UID", "SHAPE@LENGTH"))
        iddisc = {}
        for row in rows:
            #print row[0]
            uniqueid = int(re.split('_', row[1])[0])
            unqid = int(re.split('_', row[1])[1])
            shpln = float(row[2])
            if not uniqueid in iddisc:
                iddisc[uniqueid] = list()
            iddisc[uniqueid].append((unqid, shpln))
            del row
        del rows

        # Begining XSCL
        w = open(temp_dbf, 'w')
        w.writelines("FNXSCL,XSCLUID,LN\n")
        for k in iddisc.keys():
            t = list()
            for v in iddisc[k]:
                t.append(v[0])
            buid = str(k)+"_"+str(max(t))
            for jj in iddisc[k]:
                if (buid == str(k)+"_"+str(jj[0])):
                    if (float(jj[1]) < float(T)):
                        w.writelines(str(9999)+","+str(buid)+","+str(jj[1])+"\n")
                del jj
            del k
        w.close()    
        del iddisc

        arcpy.AddMessage("Join Field")
        arcpy.CopyRows_management(temp_dbf, j_table)
        arcpy.JoinField_management(in_xscl, "XSCL_UID", j_table, "XSCLUID", "FNXSCL")##
        arcpy.FeatureClassToFeatureClass_conversion(in_xscl, os.path.dirname(model_output_18_str), str(model_output_18_str.split("\\")[-1]), "\"FNXSCL\" = 9999", "", "")

        if os.path.exists(temp_dbf):
            os.remove(temp_dbf)
        if arcpy.Exists(j_table):
            arcpy.Delete_management(j_table, "")

        # Update field
        arcpy.AddMessage("Update Field")
        fld = arcpy.ListFields(model_output_18_str)
        for f in fld:
            if f.name == "Comments":
                arcpy.CalculateField_management(model_output_18_str, "Comments", "\"First Up-stream Cross-section is LT "+str(T)+" m\"", "PYTHON_9.3", "")
                pass
            else:
                arcpy.AddField_management(model_output_18_str, "Comments", "TEXT", "", "", 255)
                arcpy.CalculateField_management(model_output_18_str, "Comments", "\"First Up-stream Cross-section is LT "+str(T)+" m\"", "PYTHON_9.3", "")
            del f
        del fld

        # Delete field from input file
        fld12 = arcpy.ListFields(in_xscl)
        for ii in fld12:
            if ii.name == "XSCLUID":
                arcpy.DeleteField_management(in_xscl, ["XSCLUID"])
            elif ii.name == "FNXSCL":
                arcpy.DeleteField_management(in_xscl, ["FNXSCL"])
            else:
                pass
            del ii
        del fld12

        # Delete field from output file
        fld13 = arcpy.ListFields(model_output_18_str)
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
                arcpy.DeleteField_management(model_output_18_str, [str(ii.name)])
            del ii
        del fld13

    except Exception as e:
        print (e.message)
        arcpy.AddMessage(e.message)

    return

if __name__ == '__main__':
    XSCL_QAQC_Layer_J(Workspace, inCrosssection, UXL, qaqcOutput)
