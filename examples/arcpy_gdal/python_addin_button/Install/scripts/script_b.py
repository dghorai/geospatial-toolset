# Import modules
import re
import os
import sys
import numpy
import arcpy

arcpy.env.overwriteOutput = True

# Input arguments
Workspace = arcpy.GetParameterAsText(0)
inCrosssection = arcpy.GetParameterAsText(1)
TH = float(arcpy.GetParameterAsText(2)) #75 # Threshols - ((XS_length - Avg_length)/Avg_length)*100
modelOutput = arcpy.GetParameterAsText(3)

# Function for XSCL QAQC Laye C
def XSCL_QAQC_Layer_K(workspace, in_xscl, T, out2):
    try:
        print ("1 : XSCL is Inconsistance")
        arcpy.AddMessage("XSCL is Inconsistance")

        # Delete files (if exists)
        if arcpy.Exists(out2):
            arcpy.Delete_management(out2, "")

        # Find xscl along the single flowline which is not linear by length (say, 75% high/low from avg length)
        #----------------------------------------------------------------------------
        try:
            rows = arcpy.da.SearchCursor(in_xscl, ('XSCL_UID', 'SHAPE@LENGTH'))
            iddisc = {}
            for row in rows:
                uniqueid = int(re.split('_', row[0])[0])
                unqid = int(re.split('_', row[0])[1])
                shpln = float(row[1])
                if not uniqueid in iddisc:
                    iddisc[uniqueid] = list()
                iddisc[uniqueid].append((unqid, shpln))
                del row
            del rows
        except:
            sys.exit("Check the XSCL_UID!")

        # Get the update cursor
        folder = os.path.dirname(workspace)
        inc_txt = folder+"\\"+"inc_txt.txt"
        j_table = workspace+"\\"+"j_table"
        if os.path.exists(inc_txt):
            os.remove(inc_txt)
            inc_txt = folder+"\\"+"inc_txt.txt"
            
        w = open(inc_txt, 'w')
        w.writelines("XSCLUID,XS_INC\n")
        for k in iddisc.keys():
            vlist = list()
            for v in iddisc[k]:
                vlist.append(v[1])
            avg = numpy.mean(vlist)
            for v1 in iddisc[k]:
                if v1[1] > avg:
                    P = (((v1[1]-avg)/avg)*100)
                    if (P > T):
                        w.writelines(str(k)+"_"+str(v1[0])+","+str(P)+"\n")
                if v1[1] < avg:
                    M = (((v1[1]-avg)/avg)*100)
                    if (M < (-T)):
                        w.writelines(str(k)+"_"+str(v1[0])+","+str(P)+"\n")
                del v1
            del k
        w.close()

        # Attribute join
        arcpy.AddMessage("Join Field")
        arcpy.CopyRows_management(inc_txt, j_table)
        arcpy.JoinField_management(in_xscl, "XSCL_UID", j_table, "XSCLUID", "XS_INC") #####
        
        # XS selection GT T
        xs_inc = arcpy.MakeFeatureLayer_management(in_xscl, 'xsinclyr', "", workspace, "")
        arcpy.SelectLayerByAttribute_management(xs_inc, "NEW_SELECTION", "\"XS_INC\" IS NULL OR \"XS_INC\" =0")###
        arcpy.SelectLayerByAttribute_management(xs_inc, "SWITCH_SELECTION", "")
        arcpy.CopyFeatures_management(xs_inc, out2, "", "0", "0", "0")
        arcpy.SelectLayerByAttribute_management(xs_inc, "CLEAR_SELECTION")
        arcpy.Delete_management(xs_inc, "")

        # Update field
        arcpy.AddMessage("Update Field")
        fld = arcpy.ListFields(out2)
        for f in fld:
            if f.name == "Comments":
                arcpy.CalculateField_management(out2, "Comments", "\"Cross-section is inconsistance\"", "PYTHON_9.3", "")
                pass
            else:
                arcpy.AddField_management(out2, "Comments", "TEXT", "", "", 255)
                arcpy.CalculateField_management(out2, "Comments", "\"Cross-section is inconsistance\"", "PYTHON_9.3", "")
            del f
        del fld

        # Delete field from input file
        fld12 = arcpy.ListFields(in_xscl)
        for ii in fld12:
            if ii.name == "XSCLUID":
                arcpy.DeleteField_management(in_xscl, ["XSCLUID"])
            elif ii.name == "XS_INC":
                arcpy.DeleteField_management(in_xscl, ["XS_INC"])
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

        if os.path.exists(inc_txt):
            os.remove(inc_txt)
        if arcpy.Exists(j_table):
            arcpy.Delete_management(j_table, "")

    except Exception as e:
        print (e.message)
        arcpy.AddMessage(e.message)
    
    return

# Inputs/output
if __name__ == '__main__':
    XSCL_QAQC_Layer_K(Workspace, inCrosssection, TH, modelOutput)
