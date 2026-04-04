# Polygon to line conversion
# Scripted by Debabrata
# Import modules
import arcpy
arcpy.env.overwriteOutput = True

# Function for polygon to polyline conversion
def Polygon_to_Line(InputPoly, Pnt2Line):
    # Polygon to line conversion
    spatial_reference = arcpy.Describe(InputPoly).spatialReference
    desc = arcpy.Describe(InputPoly)
    shapefieldname = desc.ShapeFieldName
    rows = arcpy.SearchCursor(InputPoly)
    featList = list()
    features = list()
    cnt = 1
    for row in rows:
        #print "Polygon Feature No. "+str(cnt)
        feat = row.getValue(shapefieldname)
        partnum = 0
        ftlist = list()
        for part in feat:
            for pnt in feat.getPart(partnum):
                if pnt:
                    ftlist.append([pnt.X, pnt.Y])
                else:
                    ir = "Interior ring"
        featList.append(ftlist)
        cnt += 1
    del rows
    for feature in featList:
        features.append(
            arcpy.Polyline(
                arcpy.Array([arcpy.Point(*i) for i in feature])))
    del featList
    #print "Copy polyline file"
    arcpy.CopyFeatures_management(features, Pnt2Line)
    arcpy.DefineProjection_management(Pnt2Line, spatial_reference)
    del features

if __name__ == '__main__':
    import sys
    Polygon_to_Line(sys.argv[1], sys.argv[2])
