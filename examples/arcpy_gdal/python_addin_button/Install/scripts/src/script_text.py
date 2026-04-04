# Import modules
import os
import arcpy
arcpy.env.overwriteOutput = True

def Min_At_End(inputfile1, outputfile1):
    inputfile = open(inputfile1, 'r')
    outputfile = open(outputfile1, 'w')
    readLine = False
    outputfile.write("XSCLUID,ST_ELV,ED_ELV,MIN_ELV,UVAL\n")
    for line in inputfile:
        try:
            # Start with "XS"
            if line.startswith("XS"):
                readLine = True
                elvProfile = list()
                distProfile = list() # Not required
                profileName = line.split(' ')[1]
                
            # Start with "*"
            elif readLine and line.startswith("*"):
                readLine = False
                # Get minimum elevation from "readline (below)" created list for each "XS (above)"
                minelv = min(elvProfile)
                startelv = elvProfile[0]
                endelv = elvProfile[len(elvProfile)-1]
                if startelv == minelv or endelv == minelv:
                    outputfile.write(profileName+","+str(startelv)+","+str(endelv)+","+str(minelv)+","+str(9999)+"\n")
                    
            # Start with "dist elv" values
            elif readLine:
                elvProfile.append([float(ii) for ii in line.split(' ')][1])
                distProfile.append([float(ii) for ii in line.split(' ')][0]) # Not required
                
        except Exception as e:
            print (e.message)
            arcpy.AddMessage(e.message)
            
        del line
    inputfile.close()
    outputfile.close()

    return

def Min_From_Textfile(workspace, intext, inxscl, outxscl):
    try:
        #outxscl = workspace+"\\"+"output_min_text"
        if arcpy.Exists(outxscl):
            arcpy.Delete_management(outxscl, "")
            
        outtxt = os.path.dirname(intext)+"\\"+"min_at_end.txt"
        j_table = os.path.dirname(outxscl)+"\\"+"j_table"
        if arcpy.Exists(j_table):
            arcpy.Delete_management(j_table, "")
        Min_At_End(intext, outtxt)
        arcpy.CopyRows_management(outtxt, j_table)
        os.remove(outtxt)
        arcpy.AddMessage("Join Field")
        fld11 = arcpy.ListFields(inxscl)
        for jj in fld11:
            if jj.name == "UVAL":
                arcpy.DeleteField_management(inxscl, ["UVAL"])
            del jj
        del fld11
        arcpy.JoinField_management(inxscl, "XSCL_UID", j_table, "XSCLUID", "UVAL")
        arcpy.FeatureClassToFeatureClass_conversion(inxscl, os.path.dirname(outxscl), str(outxscl.split("\\")[-1]), "\"UVAL\" = 9999", "", "")
        if arcpy.Exists(j_table):
            arcpy.Delete_management(j_table, "")
        # Update field
        arcpy.AddMessage("Update Field")
        fld10 = arcpy.ListFields(outxscl)
        for f in fld10:
            if f.name == "Comments":
                arcpy.CalculateField_management(outxscl, "Comments", "\"Minimum Value at End\"", "PYTHON_9.3", "")
                pass
            else:
                arcpy.AddField_management(outxscl, "Comments", "TEXT", "", "", 255)
                arcpy.CalculateField_management(outxscl, "Comments", "\"Minimum Value at End\"", "PYTHON_9.3", "")
            del f
        del fld10
        # Delete field from input file
        fld12 = arcpy.ListFields(inxscl)
        for ii in fld12:
            if ii.name == "UVAL":
                arcpy.DeleteField_management(inxscl, ["UVAL"])
            del ii
        del fld12
        # Delete field from output file
        fld13 = arcpy.ListFields(outxscl)
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
                arcpy.DeleteField_management(outxscl, [str(ii.name)])
            del ii
        del fld13
    except Exception as e:
        arcpy.AddMessage(e.message)

    return

if __name__ == '__main__':
    import sys
    Min_From_Textfile(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
