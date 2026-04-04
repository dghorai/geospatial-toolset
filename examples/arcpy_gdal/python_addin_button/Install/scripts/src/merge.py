# Function for merge feature class, identify duplicate and update the records
# Import modules
import os
import arcpy
arcpy.env.overwriteOutput = True

# Function for merge, duplicate and update records
def Duplicate_Geometry(fc):
    drows = arcpy.da.SearchCursor(fc, ('SHAPE@', 'DGCOMID'))
    glist = list()
    for g in drows:
        glist.append((g[0], g[1]))
        del g
    del drows
    # Loop over the geometry
    objdisc = {}
    for g1 in glist:
        for g2 in glist:
            if g1[0].equals(g2[0]) == True:
                if not g1[1] in objdisc:
                    objdisc[g1[1]] = list()
                objdisc[g1[1]].append(g2[1])
            del g2
        del g1
    del glist
    objtemp = list()
    for i in objdisc.keys():
        if len(objdisc[i]) > 1:
            objdisc[i].sort()
            objtemp.append(tuple(objdisc[i])) # Convert list to tuple (since list within list can not be perform list(set()) operation)
        del i
    del objdisc
    # Eliminate duplicate from the tuple-list
    objdup = list(set(objtemp))
    del objtemp
    return objdup

def Duplicate_Values(cursor, duplc):
    dp = list()
    for rr in cursor:
        for jj in duplc:
            if (int(rr[0]) == int(jj)):
                dp.append(rr[1])
            del jj
        del rr
    #print len(dp), list(set(dp)) 
    return list(set(dp)) # Fixed the issue

def Update_Cursor(fclist, cmlist, output3):
    try:
        for f, m in zip(fclist, cmlist):
            arcpy.AddField_management(f, "Comments", "TEXT", "", "", 255)
            arcpy.CalculateField_management(f, "Comments", "\""+m+"\"", "PYTHON_9.3", "")
        arcpy.Merge_management(fclist, output3)
        # Create a DGCOMID
        f123 = arcpy.ListFields(output3)
        for fi in f123:
            if fi.name == "DGCOMID":
                pass
            else:
                arcpy.AddField_management(output3, "DGCOMID", "DOUBLE", "", "", "", "", "NULLABLE", "NON_REQUIRED", "")
        #arcpy.AddMessage("Row Count")
        row123 = arcpy.da.UpdateCursor(output3, ["DGCOMID"])
        rcnt = 1
        for row in row123:
            row[0] = rcnt
            row123.updateRow(row)
            rcnt += 1
            del row
        # Get individual error counts
        ctble = os.path.dirname(output3)+"\\"+"Table_"+output3.split("\\")[-1]
        arcpy.AddMessage("Output Table : "+str(ctble))
        fldr1 = os.path.dirname(output3)
        fldr2 = os.path.dirname(fldr1)+"\\"+"Table_"+output3.split("\\")[-1]+".txt"
        if os.path.exists(fldr2):
            os.remove(fldr2)
            fldr2 = os.path.dirname(fldr1)+"\\"+"Table_"+output3.split("\\")[-1]+".txt"
        wr = open(fldr2, 'w')
        wr.writelines("Type,Count\n")
        for fc, cm in zip(fclist, cmlist):
            getcnt = arcpy.GetCount_management(fc)
            wr.writelines(cm+","+str(getcnt)+"\n")
        wr.close()
        arcpy.CopyRows_management(fldr2, ctble)
        if os.path.exists(fldr2):
            os.remove(fldr2)
        for i in fclist:
            arcpy.Delete_management(i, "")
        arcpy.AddMessage("Duplicate Rows")
        duplicate = Duplicate_Geometry(output3)
        # Update featureclass
        arcpy.AddMessage("Update Fields")
        dellist = list()
        for d in duplicate:
            rows = arcpy.da.UpdateCursor(output3, ('DGCOMID', 'Comments'))
            for row in rows:
                #arcpy.AddMessage((row[0], d[0]))
                if (int(row[0]) == int(d[0])):
                    rows1 = arcpy.da.SearchCursor(output3, ('DGCOMID', 'Comments'))
                    dv = Duplicate_Values(rows1, d)
                    del rows1
                    row[1] = ",".join(str(ii) for ii in dv)
                    rows.updateRow(row)
                    del dv
                    for dp in d[1:]:
                        dellist.append(dp)
                        del dp
                del row
            del rows
            del d
        rows1 = arcpy.UpdateCursor(output3)
        for cur in rows1:
            for jj in dellist:
                if (int(cur.DGCOMID) == int(jj)):
                    rows1.deleteRow(cur)
                del jj
            del cur
        del rows1
        del dellist
    except Exception as e:
        print (e.message)
        arcpy.AddMessage(e.message)

# Call the function
if __name__ == '__main__':
    import sys
    Update_Cursor(sys.argv[1], sys.argv[2], sys.argv[3])
    
