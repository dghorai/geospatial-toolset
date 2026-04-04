# Import Modules
import os
import arcpy
arcpy.env.overwriteOutput = True

def Filter_Polyline_FC(infc, outfc):
    try:
        # Line to Both-End Points
        pntdisc = {}
        for row in arcpy.da.SearchCursor(infc, ["OBJECTID", "SHAPE@"]):
            startpt = row[1].firstPoint
            endpt = row[1].lastPoint
            sxy = (float(format(startpt.X, '.6f')), float(format(startpt.Y, '.6f')))
            exy = (float(format(endpt.X, '.6f')), float(format(endpt.Y, '.6f')))
            totxy = [sxy, exy]
            for ii in totxy:
                if not ii in pntdisc:
                    pntdisc[ii] = list()
                pntdisc[ii].append(ii)
                del ii
            del row

        # Get duplicate/common points
        ptfile = os.path.dirname(infc)+"\\"+"point_file123"
        if arcpy.Exists(ptfile):
            arcpy.Delete_management(ptfile, "")
        sr = arcpy.Describe(infc).spatialReference
        ptfc = arcpy.CreateFeatureclass_management(os.path.dirname(infc), "point_file123", 'POINT','','','',sr)
        icursor = arcpy.da.InsertCursor(ptfc, ("SHAPE@"))
        for k in pntdisc.keys():
            ln = len(pntdisc[k])
            if ln > 1:
                icursor.insertRow((k,))
            del k
        del icursor
        del pntdisc

        # Select Line only for single point
        lyrfile = arcpy.MakeFeatureLayer_management(infc, "infc_Layer", "", "", "")
        arcpy.SelectLayerByLocation_management(lyrfile, "INTERSECT", ptfile, "0.1 Meters", "NEW_SELECTION")
        arcpy.SelectLayerByAttribute_management(lyrfile, "SWITCH_SELECTION", "")
        arcpy.CopyFeatures_management(lyrfile, outfc, "", "0", "0", "0")
        arcpy.SelectLayerByAttribute_management(lyrfile, "CLEAR_SELECTION")
        arcpy.Delete_management(lyrfile, "")
        arcpy.Delete_management(ptfile, "")
    except Exception as e:
        arcpy.AddMessage(e.message)

    return

if __name__ == '__main__':
    import sys
    Filter_Polyline_FC(sys.argv[1], sys.argv[2])
    
