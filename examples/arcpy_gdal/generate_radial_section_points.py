## radian section points generation using arcpy module
import arcpy
import sys, math
from itertools import groupby
from operator import itemgetter
from arcpy import env
env.overwriteOutput = True

# Local variables
OutDir = arcpy.GetParameterAsText(0)
if OutDir == "#" or not OutDir:
    OutDir = r"C:\Work\Temporary.gdb"
inFeatures = arcpy.GetParameterAsText(1)
if inFeatures == "#" or not inFeatures:
    inFeatures = r"C:\Work\Data.gdb\TestData"
outFC = OutDir+"/"+"splitline"
alongDist = arcpy.GetParameterAsText(2)
if alongDist == "#" or not alongDist:
    alongDist = 1000
CoordinateSystem = arcpy.GetParameterAsText(3)
if CoordinateSystem == "#" or not CoordinateSystem:
    CoordinateSystem = "PROJCS['Asia_North_Albers_Equal_Area_Conic',GEOGCS['GCS_WGS_1984',DATUM['D_WGS_1984',SPHEROID['WGS_1984',6378137.0,298.257223563]],PRIMEM['Greenwich',0.0],UNIT['Degree',0.0174532925199433]],PROJECTION['Albers'],PARAMETER['false_easting',0.0],PARAMETER['false_northing',0.0],PARAMETER['central_meridian',95.0],PARAMETER['standard_parallel_1',15.0],PARAMETER['standard_parallel_2',65.0],PARAMETER['latitude_of_origin',30.0],UNIT['Meter',1.0]];-16934700 -8092500 10000;-100000 10000;-100000 10000;0.001;0.001;0.001;IsHighPrecision"

outputFile = OutDir+"/"+"FlipLine"
pointFile = OutDir+"/"+"RSPoints"
RSPointsJoin = OutDir+"/"+"RadialSectionPoints"
Feature_Dataset = OutDir+"/"+"Feature_Dataset"
RSP = OutDir+"/"+"RSP"
RSPoints_Layer = "RSPoints_Layer"
Feature_Dataset_Net_Junctions = Feature_Dataset+"\\Feature_Dataset_Net_Junctions"
Feature_Dataset_Net_Junction = "Feature_Dataset_Net_Junction"
FinalRSP = OutDir+"/"+"FinalRSP"

# Processing
print("Flip lines")
inFC = arcpy.CopyFeatures_management(inFeatures, outputFile, "", "", "")
arcpy.FlipLine_edit(inFC)

# File existance checking
if (arcpy.Exists(inFC)):
    print(str(inFC)+" does exist")
    arcpy.AddMessage(str(inFC)+" does exist")
else:
    print("Cancelling, "+str(inFC)+" does not exist")
    arcpy.AddMessage("Cancelling, "+str(inFC)+" does not exist")
    sys.exit(0)

# Spliting lines into set interval
def distPoint(p1, p2):
    calc1 = p1.X - p2.X
    calc2 = p1.Y - p2.Y
    return math.sqrt((calc1**2)+(calc2**2))

def midpoint(prevpoint,nextpoint,targetDist,totalDist):
    newX = prevpoint.X + ((nextpoint.X - prevpoint.X) * (targetDist/totalDist))
    newY = prevpoint.Y + ((nextpoint.Y - prevpoint.Y) * (targetDist/totalDist))
    return arcpy.Point(newX, newY)

def splitShape(feat,splitDist):
    partcount = feat.partCount
    partnum = 0
    lineArray = arcpy.Array()
    while partnum < partcount:
        part = feat.getPart(partnum)
        totalDist = 0
        pnt = part.next()
        pntcount = 0
        prevpoint = None
        shapelist = []
        while pnt:
            if not (prevpoint is None):
                thisDist = distPoint(prevpoint,pnt)
                maxAdditionalDist = splitDist - totalDist
                print("Dist-", thisDist, totalDist, maxAdditionalDist)
                if (totalDist+thisDist)> splitDist:
                    while(totalDist+thisDist) > splitDist:
                        maxAdditionalDist = splitDist - totalDist
                        print("Dist-", thisDist, totalDist, maxAdditionalDist)
                        newpoint = midpoint(prevpoint,pnt,maxAdditionalDist,thisDist)
                        lineArray.add(newpoint)
                        shapelist.append(lineArray)
                        lineArray = arcpy.Array()
                        lineArray.add(newpoint)
                        prevpoint = newpoint
                        thisDist = distPoint(prevpoint,pnt)
                        totalDist = 0
                    lineArray.add(pnt)
                    totalDist+=thisDist
                else:
                    totalDist+=thisDist
                    lineArray.add(pnt)
            else:
                lineArray.add(pnt)
                totalDist = 0
            prevpoint = pnt                
            pntcount += 1
            pnt = part.next()
            if not pnt:
                pnt = part.next()
                if pnt:
                    print("Interior Ring:")
                    arcpy.AddMessage("Interior Ring:")
        partnum += 1
    if (lineArray.count > 1):
        shapelist.append(lineArray)
    return shapelist

if arcpy.Exists(outFC):
    arcpy.Delete_management(outFC)
    
arcpy.Copy_management(inFC,outFC)
deleterows = arcpy.UpdateCursor(outFC)

for iDRow in deleterows:       
     deleterows.deleteRow(iDRow)
     
del iDRow
del deleterows

inputRows = arcpy.SearchCursor(inFC)
outputRows = arcpy.InsertCursor(outFC)
fields = arcpy.ListFields(inFC)
numRecords = int(arcpy.GetCount_management(inFC).getOutput(0))
OnePercentThreshold = numRecords/100
print("Lines-", numRecords)
arcpy.AddMessage("Lines: " + str(numRecords))

iCounter = 0
iCounter2 = 0

for iInRow in inputRows:
    inGeom = iInRow.shape
    iCounter+=1
    iCounter2+=1    
    if (iCounter2 > (OnePercentThreshold+0)):
        print("Processing Record "+str(iCounter) + " of "+ str(numRecords))
        arcpy.AddMessage("Processing Record "+str(iCounter) + " of "+ str(numRecords))
        iCounter2=0
    if (inGeom.length > alongDist):
        shapeList = splitShape(iInRow.shape,alongDist)
        for itmp in shapeList:
            newRow = outputRows.newRow()
            for ifield in fields:
                if (ifield.editable):
                    newRow.setValue(ifield.name,iInRow.getValue(ifield.name))
            newRow.shape = itmp
            outputRows.insertRow(newRow)
    else:
        outputRows.insertRow(iInRow)
        
del inputRows
del outputRows

# Segments nodes extracting
inputFC = outFC
desc = arcpy.Describe(inputFC)
shapefieldname = desc.ShapeFieldName
rows = arcpy.SearchCursor(inputFC)
pointList = []
UNIQUEID = []
for row in rows:
    UNIQUEID = row.UNIQUEID
    feat = row.getValue(shapefieldname)
    oid = row.getValue(desc.OIDFieldName)
    print("Feature %i:" % oid)
    partnum = 0
    for part in feat:
        part = "Part %i:" % partnum
        print(part)
        segPointList = []
        for pnt in feat.getPart(partnum): 
            if pnt:
                x = pnt.X
                y = pnt.Y
                lst = [x, y]
                segPointList.append(lst)
            else:
                print("Interior Ring:")
        pointList.append(segPointList)
        partnum += 1

# Gathering end nodes of segments into a list
ptList = []
for i in range(len(pointList)):
    print("Node Number: "+str(i))
    startNode = pointList[i][-1]
    x1 = float(startNode[0])
    y1 = float(startNode[1])
    pnt = [x1,y1]
    ptList.append(pnt)

# Point file creation of end nodes
pt = arcpy.Point()
ptGeoms = []
for p in ptList:
    pt.X = p[0]
    pt.Y = p[1]
    ptGeoms.append(arcpy.PointGeometry(pt))

arcpy.CopyFeatures_management(ptGeoms, pointFile)

# Junction point removing
print("Create Feature Dataset")
arcpy.CreateFeatureDataset_management(OutDir, "Feature_Dataset", CoordinateSystem)
print("Copy Feature Class")
arcpy.FeatureClassToFeatureClass_conversion(inFeatures, Feature_Dataset, "RSP", "", "", "")
print("Create Geometric Network")
arcpy.CreateGeometricNetwork_management(Feature_Dataset, "Feature_Dataset_Net", "RSP SIMPLE_EDGE NO", "", "", "", "", "PRESERVE_ENABLED")
print("Make Feature Layer-Point")
arcpy.MakeFeatureLayer_management(pointFile, RSPoints_Layer, "", "", "OBJECTID OBJECTID VISIBLE NONE;Shape Shape VISIBLE NONE")
print("Make Feature Layer-Point Junction")
arcpy.MakeFeatureLayer_management(Feature_Dataset_Net_Junctions, Feature_Dataset_Net_Junction, "", "", "OBJECTID OBJECTID VISIBLE NONE;SHAPE SHAPE VISIBLE NONE;Enabled Enabled VISIBLE NONE")
print("Query By Location")
arcpy.SelectLayerByLocation_management(RSPoints_Layer, "INTERSECT", Feature_Dataset_Net_Junction, "20 Meters", "NEW_SELECTION")
print("Query By Attribute")
arcpy.SelectLayerByAttribute_management(RSPoints_Layer, "SWITCH_SELECTION", "")
print("Export Feature Class")
arcpy.FeatureClassToFeatureClass_conversion(RSPoints_Layer, OutDir, "FinalRSP", "", "", "")

# Spatial Join
print("Spatial Join")
arcpy.SpatialJoin_analysis(FinalRSP, inFeatures, RSPointsJoin, "JOIN_ONE_TO_ONE", "KEEP_COMMON", "UNIQUEID \"UNIQUEID\" true true false 4 Long 0 0 ,First,#,"+inFeatures+",UNIQUEID,-1,-1", "CLOSEST", "0.0001 Meters", "")

# Add Field
arcpy.AddField_management(RSPointsJoin, "XSCL_ID", "TEXT", "", "", "", "", "NULLABLE", "NON_REQUIRED", "")
arcpy.AddField_management(RSPointsJoin, "DISTANCE", "TEXT", "", "", "", "", "NULLABLE", "NON_REQUIRED", "")

# UID Generation
rows = arcpy.SearchCursor(RSPointsJoin)
objectID = []
UNIQUEID = []
xsclID = []
for row in rows:
    objectid = row.OBJECTID
    UNIQUEID = row.UNIQUEID
    xsclid = [UNIQUEID, objectid]
    objectID.append(objectid)
    UNIQUEID.append(UNIQUEID)
    xsclID.append(xsclid)
    
combined = [(k, [v[1] for v in g]) for k, g in
            groupby(sorted(xsclID), key=itemgetter(0))]

uidList = []
for i in range(len(combined)):
    air = combined[i][0]
    print("UNIQUEID: "+ str(air))
    xscl = combined[i][1]
    for j in range(len(xscl)):
        uid = str(air)+"_"+str(j)
        text = [str(xscl[j]), str(uid), str((j+1)*90)]
        uidList.append(text)

# Update feature class row by values
cur = arcpy.UpdateCursor(RSPointsJoin)
for row in cur:
    for i in range(len(uidList)):
        value = uidList[i]
        if value[0] == str(row.OBJECTID):
            row.setValue("XSCL_ID", value[1])
            row.setValue("DISTANCE", value[2])
            cur.updateRow(row)

del cur
print("Done!")
