# Get Hourly Wind of Cyclone Track Points

"""
Get Hourly Wind of Cyclone Track Points
Date: 2016-09-07
Platform: Python 2.7 and ArcPy
"""

# Import modules
import os
import math
import arcpy

arcpy.env.overwriteOutput = True

# Input arguments
inputCyTracks = r"C:\Work\Inputs.gdb\point_file"
hourSequenceID = "ID"
hourField = "HOUR"
windField = "WIND"
newFileName = "Point_new"
outWindField = "W"

# Function for distance calculation
def DistCalc(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

# Function for hourly node generation
def HourWindNode(hrDiff, w, wDiff):
    nodeList = list()
    line = arcpy.Polyline(arcpy.Array([arcpy.Point(w[0][1][0], w[0][1][1]), arcpy.Point(w[1][1][0], w[1][1][1])]))
    node = (w[0][1], w[1][1])
    intv = float(DistCalc(node[0], node[1])/hrDiff)
    for p in range(hrDiff):
        pnt = line.positionAlongLine(intv*p).firstPoint
        nodeList.append((pnt.X, pnt.Y, w[0][0]+wDiff*p))
    return nodeList

# Function for wind value calculation an hourly
def HourlyWindValue(infile, hour_seq_id, hour, wind, outfilename, out_wind_fld):
    try:
        # Read file
        rows = arcpy.da.SearchCursor(infile, ["SHAPE@XY", hour_seq_id, hour, wind])
        pointList = list()
        slridList = list()
        for row in rows:
            pointList.append(row)
            slridList.append(int(row[1]))
            del row
        slridList.sort()

        # Point pair construction
        pntPair = list()
        for i in range(len(slridList)-1):
            pntPair.append(slridList[i:i+2])
            del i

        # Loop over the point-pair
        wp = list()
        for i in pntPair:
            hrList = list()
            wList = list()
            for j in i:
                for k in pointList:
                    if j == int(k[1]):
                        hrList.append(int(k[2]))
                        wList.append((float(k[3]),k[0]))
                    del k
                del j
            if hrList[0] < hrList[1]:
                hrDiff = (hrList[1] - hrList[0])/100
                wDiff = float((wList[1][0] - wList[0][0])/hrDiff)
                txt = HourWindNode(hrDiff, wList, wDiff)
                for t in txt:
                    wp.append(t)
                    del t
                del txt
            else:
                if hrList[1] == 0:
                    hrDiff = (2400 - hrList[0])/100
                    wDiff = float((wList[1][0] - wList[0][0])/hrDiff)
                    txt = HourWindNode(hrDiff, wList, wDiff)
                    for t in txt:
                        wp.append(t)
                        del t
                    del txt
                else:
                    print ("Check the hour value!")
            del i
        # Append the end point value
        ep = slridList[-1]
        for lp in pointList:
            if int(lp[1]) == ep:
                wp.append((lp[0][0], lp[0][1], float(lp[3])))
            del lp
        del pointList
        del slridList

        # Point feature class
        sr = arcpy.Describe(infile).spatialReference
        segPts = arcpy.CreateFeatureclass_management(os.path.dirname(infile), outfilename, 'POINT','','','',sr)
        arcpy.AddField_management(segPts, out_wind_fld, "DOUBLE", "", "", "", "", "NULLABLE", "REQUIRED")
        icursor = arcpy.da.InsertCursor(segPts, ("SHAPE@", out_wind_fld))
        for r in wp:
            icursor.insertRow(((r[0], r[1]),r[2]))
            del r
        del icursor
        del wp

    except Exception, e:
        print (e.mesage)

    return

# Input/output files
if __name__ == '__main__':
    HourlyWindValue(inputCyTracks, hourSequenceID, hourField, \
                    windField, newFileName, outWindField)
    print ("Done!")
