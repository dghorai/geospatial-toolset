# Import arcpy module
import os
import math
import arcpy
arcpy.env.overwriteOutput = True

def DistCalc(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def MFL_Crosses_Filter(inmflflags, inboundary, outputflags):
    try:
        pntoutput = os.path.dirname(outputflags)+"\\"+"temp_points_file"
        if arcpy.Exists(pntoutput):
            arcpy.Delete_management(pntoutput, "")

        fields = arcpy.ListFields(inmflflags)
        for ii in fields:
            if ii.name == "COMID":
                pass
            elif ii.name == "FLG":
                pass
            else:
                arcpy.AddField_management(inmflflags, "COMID", "LONG", "", "", "", "", "NULLABLE", "REQUIRED")
                arcpy.AddField_management(inmflflags, "FLG", "LONG", "", "", "", "", "NULLABLE", "REQUIRED")
            del ii
        del fields

        nodelist = list()
        cursor = arcpy.da.UpdateCursor(inmflflags, ["COMID", "SHAPE@"])
        for cnt, row in enumerate(cursor):
            row[0] = cnt+1
            cursor.updateRow(row)
            pt = row[1].lastPoint
            nodelist.append((row[0], (pt.X, pt.Y)))
            del row
        del cursor

        arcpy.Intersect_analysis(inboundary+" #;"+inmflflags+" #", pntoutput, "ALL", "0.001 Meters", "POINT")

        dist = None
        flglist = list()
        for pp in arcpy.da.SearchCursor(pntoutput, ["COMID", "SHAPE@XY"]):
            pnt = (pp[1][0], pp[1][1])
            for jj in nodelist:
                if pp[0] == jj[0]:
                    dist = float(format(DistCalc(pnt, jj[1]), '0.2f'))
            if dist > 0.00:
                flglist.append(pp[0])

        rows = arcpy.da.UpdateCursor(inmflflags, ["COMID", "FLG"])
        for ii in rows:
            for jj in flglist:
                if ii[0] == jj:
                    ii[1] = 9999
            rows.updateRow(ii)
        del rows

        arcpy.FeatureClassToFeatureClass_conversion(inmflflags, os.path.dirname(outputflags), str(outputflags.split("\\")[-1]), "\"FLG\" = 9999", "", "")

        if arcpy.Exists(inmflflags):
            arcpy.Delete_management(inmflflags, "")
        if arcpy.Exists(pntoutput):
            arcpy.Delete_management(pntoutput, "")

    except Exception as e:
        arcpy.AddMessage(e.message)

    return

if __name__ == '__main__':
    import sys
    MFL_Crosses_Filter(sys.argv[1], sys.argv[2], sys.argv[3])
