# HYDROBURN RIVER CHANNEL

# Developed By : Debabrata Ghorai
# Developed Date : 14-11-2017

# Steps:
# 1) Create river points, assign point's Unique-ID, and extract DEM elevation to point
# 2) Calculate median elevation for each point based on user define Point-Window-Size 
# 3) Filter high elevated median values of a river point and replace that point's elevation by following or preceding point elevation which is near on the river

# Before RUN the script just check the presence of "UNIQUEID" field in river flowline feature class


# Import modules
import os
import time
import math
import numpy
import arcpy
import datetime

# config env
arcpy.env.overwriteOutput = True
arcpy.CheckOutExtension("Spatial")

stime = time.time()


# Input arguments:
dz = 1 # Set the threshold of elevation difference; default is 1
Range = 10 # Set the Up-Stream Point Window Size for Median; default is 10
WorkingFolder = r"D:\Projects_Working\Workspace_2.gdb"
InRiver = r"D:\Projects_Working\Region01.gdb\ModelFlowlines"
InDem = r"D:\Projects_Working\Region01.gdb\DTM_NED10M"


# PHASE-1
# Local variables
GetRiverPointAndElevation = 1 # Step-1
GetEstimatedMedianValuesZ = 2 # Step-2
GetFilteredRiverElevation = 3 # Step-3
OutRiverPoints = WorkingFolder+"\\"+str(os.path.basename(InRiver))+"_Pnt"
OutRiverPoints1 = WorkingFolder+"\\"+str(os.path.basename(OutRiverPoints))+str("s")
out_pnts = WorkingFolder+"\\"+str(os.path.basename(OutRiverPoints1))+str("_1")
MPD = 150.0 # Minimum Pixels/Points Distances
IntDem = WorkingFolder+"\\"+"ind_dem"
FloatDem = WorkingFolder+"\\"+"float_dem"
outfile = os.path.dirname(WorkingFolder)+"\\OutTable_"+str(os.path.basename(OutRiverPoints1))+".txt"
outtxt = os.path.dirname(WorkingFolder)+"\\OutTableFilter_"+str(os.path.basename(OutRiverPoints1))+".txt"

if arcpy.Exists(IntDem):
    arcpy.Delete_management(IntDem, "")
if arcpy.Exists(FloatDem):
    arcpy.Delete_management(FloatDem, "")
if arcpy.Exists(OutRiverPoints):
    arcpy.Delete_management(OutRiverPoints, "")
if arcpy.Exists(OutRiverPoints1):
    arcpy.Delete_management(OutRiverPoints1, "")

# Get Time
def timeit(ptxt, timesec):
    sec = datetime.timedelta(seconds=int(timesec))
    dty = datetime.datetime(1,1,1) + sec
    if (dty.day - 1) > 0:
        dtxt = ("%d days %d hours %d minutes %d seconds" % (dty.day-1, dty.hour, dty.minute, dty.second))
    elif dty.hour > 0:
        dtxt = ("%d hours %d minutes %d seconds" % (dty.hour, dty.minute, dty.second))
    elif dty.minute > 0:
        dtxt = ("%d minutes %d seconds" % (dty.minute, dty.second))
    else:
        dtxt = ("%d seconds" % dty.second)
    print (str(ptxt)+str(dtxt))

# Process: Feature Class Reading
def FeatureClassReading(InRivers):
    desc = arcpy.Describe(InRivers)
    shapefieldname = desc.ShapeFieldName
    rows = arcpy.SearchCursor(InRivers)
    all_nodes = []
    for row in rows:
        feat = row.getValue(shapefieldname)
        partnum = 0
        node_lst = []
        for part in feat:
            for pnt in feat.getPart(partnum):
                if pnt:
                    xy = pnt.X, pnt.Y
                    node_lst.append(xy)
                else:
                    print("Interior ring")
        all_nodes.append(node_lst)
    return all_nodes

# Process: Distance Function
def Dist(p1, p2):
    distance = math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    return distance

# Process: Near Point Identification Function
def NearPointFind(all_point, start_node):
    dist_latlong = []
    only_dist = []
    for n in all_point:
        ndist = Dist(start_node, n[0])
        dist_latlong.append((float(ndist), n))
        only_dist.append(float(ndist))

    min_dist = min(only_dist)
    for nn in dist_latlong:
        if float(nn[0]) == float(min_dist):
            near_point = nn[1]
            all_point.remove(nn[1])
            return near_point, min_dist

# Process: MIN & MAX XSCL Extraction
if GetRiverPointAndElevation == 1:
    stime1 = time.time()
    if arcpy.Raster(InDem).isInteger == False:
        print("DTM is not a integer type!")
        arcpy.gp.Int_sa(InDem, IntDem)
        arcpy.gp.Float_sa(IntDem, FloatDem)
        if arcpy.Exists(IntDem):
            arcpy.Delete_management(IntDem, "")
        etime12 = time.time()
        dtime12 = etime12 - stime1
        timeit("DTM Conversion Time : ", dtime12)
    else:
        print("DTM is integer type!")
        arcpy.CopyRaster_management(InDem, FloatDem, "", "", "-3.402823e+038", "NONE", "NONE", "", "NONE", "NONE", "", "NONE")
        etime12 = time.time()
        dtime12 = etime12 - stime1
        timeit("DTM Copy Time : ", dtime12)

    stime13 = time.time()
    mfllyr = arcpy.MakeFeatureLayer_management(InRiver, 'mfl_lyr')
    num = 1
    curr = arcpy.SearchCursor(InRiver)
    print("Reading River Channels!")

    for row in curr:
        print("\tRiver Channel : "+str(num))
        try:
            UNIQUEID = row.UNIQUEID
            expression = "\"UNIQUEID\" = "+str(row.UNIQUEID)
            arcpy.FeatureClassToFeatureClass_conversion(InRiver, WorkingFolder, "Feature_"+str(num), expression)
            SubRivFc = WorkingFolder+"\\"+"Feature_"+str(num)
            
            # Process: Extract by Mask - River
            RivRas = WorkingFolder+"\\"+"RivRaster"
            tempEnvironment0 = arcpy.env.snapRaster
            arcpy.env.snapRaster = InDem
            arcpy.gp.ExtractByMask_sa(FloatDem, SubRivFc, RivRas)
            arcpy.env.snapRaster = tempEnvironment0

            # Process: Raster to Point
            OutPoint = WorkingFolder+"\\"+"RasterPoints"
            arcpy.RasterToPoint_conversion(RivRas, OutPoint, "VALUE")

            # Process: River Start Node Acessing
            river = FeatureClassReading(SubRivFc)
            startnode = river[0][0]
            endnode = river[0][-1]

            # Process: Point Node Acessing
            dsc = arcpy.Describe(OutPoint)
            cursor = arcpy.SearchCursor(OutPoint)
            all_points = []
            allPntBackup = []
            for row in cursor:
                shape = row.getValue(dsc.shapeFieldName)
                geom = shape.getPart(0)
                xy = geom.X, geom.Y
                elev = float(row.grid_code) # Elevation Field
                all_points.append((xy, elev))
                allPntBackup.append((xy, elev))

            # Near of End_Node Point
            near_end_node = []
            allDist = []
            allDistll = []
            for i in all_points:
                endNear = Dist(endnode, i[0])
                allDist.append(endNear)
                allDistll.append((endNear,i[0]))

            md = min(allDist)
            for j in allDistll:
                if float(md) == float(j[0]):
                    near_end_node .append(j[1])
                    break

            # Process: Near Point Findings for Up-stream to Down-stream
            all_near_series = []
            for ii in range(len(all_points)):
                near = NearPointFind(all_points, startnode)
                startnode = near[0][0]
                #all_near_series.append(near)
                all_near_series.append(near[0])
                if near[0][0] == near_end_node[0]: # Calc stop at end_node
                    all_near_series.append(near[0])
                    break
                
            all_near_series2 = []
            for jj in range(len(all_near_series)-1):
                newDist = Dist(all_near_series[jj][0], all_near_series[jj+1][0])
                if newDist <= MPD:
                    all_near_series2.append(all_near_series[jj])

            # Calculate the uid for all points
            totPix = len(all_near_series2)
            cnt = totPix-1
            t = cnt
            pnt_ids = []
            for jj in all_near_series2[0:cnt]:
                pnt_ids.append((jj, str(UNIQUEID)+"_"+str(t)))
                t -= 1

            pnt_ids.append((all_near_series2[-1], str(UNIQUEID)+"_"+str(0)))

            # Point feature class creation and elevation field updating
            arcpy.CreateFeatureclass_management(WorkingFolder, "PoinFeatClass"+"_"+str(UNIQUEID), "POINT", OutPoint, "DISABLED", "DISABLED", OutPoint)
            PntFc = WorkingFolder+"\\"+"PoinFeatClass"+"_"+str(UNIQUEID)

            # Delete other fields
            fieldList = arcpy.ListFields(PntFc)
            for fd in fieldList:
                if fd.name == 'OBJECTID':
                    pass
                elif fd.name == 'Shape':
                    pass
                else:
                    arcpy.DeleteField_management(PntFc, [str(fd.name)])
            # Add fields:
            arcpy.AddField_management(PntFc, "POS_ONE", "TEXT", "", "", "", "", "NULLABLE", "NON_REQUIRED", "")
            arcpy.AddField_management(PntFc, "AIR_ID", "TEXT", "", "", "", "", "NULLABLE", "NON_REQUIRED", "")
            arcpy.AddField_management(PntFc, "POS_THREE", "TEXT", "", "", "", "", "NULLABLE", "NON_REQUIRED", "")
            arcpy.AddField_management(PntFc, "XSCL_UID", "TEXT", "", "", "", "", "NULLABLE", "NON_REQUIRED", "")
            # Update ID field
            cursor = arcpy.InsertCursor(PntFc)
            fid = 1
            for r in pnt_ids:
                vertex = arcpy.CreateObject("Point")
                vertex.X, vertex.Y = float(r[0][0][0]), float(r[0][0][1])
                feature = cursor.newRow()
                feature.shape = vertex
                feature.AIR_ID = str(r[1].split("_")[0])
                feature.XSCL_UID = str(r[1])
                cursor.insertRow(feature)
                fid += 1

            del cursor
            del all_near_series2, all_near_series
            arcpy.Delete_management(SubRivFc, "")
            arcpy.Delete_management(RivRas, "")
            arcpy.Delete_management(OutPoint, "")
        except Exception as e:
            print (e)
            
        num += 1

    # Merge
    print("\tMerge")
    arcpy.env.workspace = WorkingFolder
    fclist = arcpy.ListFeatureClasses('','')
    arcpy.Merge_management(fclist, out_pnts)

    for fc in fclist:
        arcpy.Delete_management(fc, "")
        del fc

    # Add XY
    print("\tAdding LONG and LAT to Points")
    arcpy.AddXY_management(out_pnts)

    # Extract Elevation
    print("\tExtracting DEM Elevation to Points")
    arcpy.gp.ExtractValuesToPoints_sa(out_pnts, FloatDem, OutRiverPoints1, "NONE", "VALUE_ONLY")
    arcpy.Delete_management(out_pnts, "")

    # Export text file
    print("\tExporting Text File")
    if os.path.exists(os.path.dirname(WorkingFolder)+"\\Table_"+str(os.path.basename(OutRiverPoints1))+".txt"):
        os.remove(os.path.dirname(WorkingFolder)+"\\Table_"+str(os.path.basename(OutRiverPoints1))+".txt")
        arcpy.TableToTable_conversion(OutRiverPoints1, os.path.dirname(WorkingFolder), "Table_"+str(os.path.basename(OutRiverPoints1))+".txt")
    else:
        arcpy.TableToTable_conversion(OutRiverPoints1, os.path.dirname(WorkingFolder), "Table_"+str(os.path.basename(OutRiverPoints1))+".txt")
        
    if arcpy.Exists(FloatDem):
        arcpy.Delete_management(FloatDem, "")

    etime = time.time()
    tdiff = etime - stime13
    timeit("", tdiff)
    print ("Step-1 Done!")

# PHASE-2

"""
This version of python script estimate median value for all the model flowline points. While doing this, it is appending some arbitrary values/rows
to up-stream and down-stream after the start/end point of a model flowline and then form a list of user-define range to calculate
median value to entire flowline points.
"""

# Field Description
"""
Keep File Header,
Keep Index Position:
UNIQUEID       :       [2]   ## Column Position 2
XSCL_ID     :       [4]   ## Column Position 4
X           :       [5]   ## Column Position 5
Y           :       [6]   ## Column Position 6
ELEVATION   :       [7]   ## Column Position 7
"""

# Up-stream and down-stream row management
def RowAdd(data11, Range2):
    first_num = data11[0][0]
    last_num = data11[-1][0]

    upstream_data = list()
    for bg in range(Range2):
        ups = int(first_num) - int(Range2 - bg)
        updata = ups, data11[0][1], data11[0][2], data11[0][3]
        upstream_data.append(updata)

    downstream_data = list()
    for fn in range(Range2):
        dwns = int(last_num) + int(fn+1)
        dwndata = dwns, data11[-1][1], data11[-1][2], data11[-1][3]
        downstream_data.append(dwndata)

    data1_final = upstream_data+data11+downstream_data
    return data1_final

if GetEstimatedMedianValuesZ == 2:
    stime2 = time.time()

    if os.path.exists(outfile):
        os.remove(outfile)
        outfile
    else:
        outfile

    # Local variables
    infile = os.path.dirname(WorkingFolder)+"\\Table_"+str(os.path.basename(OutRiverPoints1))+".txt"
    fopen = open(infile, 'r')
    record = list()
    comid = list()
    data1 = list()
    data2 = list()

    # Point data reading
    print("\tText Data Reading")
    for line in fopen:
        lnStrip = line.strip()
        lnSplit = lnStrip.split(",")
        # Check index for: UNIQUEID, XSCL_ID, ELEVATION, X, Y
        rcrd = lnSplit[2], lnSplit[4].split("_")[1], lnSplit[7], lnSplit[5], lnSplit[6] 
        record.append(rcrd)
        comid.append(rcrd[0])
        
    # Data with excluding header
    data = record[1:]

    # UNIQUEID making single
    UNIQUEID = list(set(comid[1:]))

    # Arrange the dataset
    print("\tDataset Arranging")
    for m in UNIQUEID:
        temp = []
        tempid = []
        tempfinal = []
        for n in data:
            if m == n[0]:
                tmp = int(n[1]), float(n[2]), float(n[3]), float(n[4])
                temp.append(tmp)
                tempid.append(tmp[0])
        tempidsort = sorted(tempid)
        for s in tempidsort:
            for ss in temp:
                if s == ss[0]:
                    # XSCL_ID Matching
                    tempfinal.append(ss)
        if len(tempfinal) > (2*Range):
            temp1 = m, tempfinal
            data1.append(temp1)
        else:
            temp2 = m, tempfinal
            data2.append(temp2)

    # Median value estimation
    print("\tMedian Calculation")
    w = open(outfile, 'w')
    w.writelines("UNIQUEID,XSCL_ID,DEM_ELV,MEDIAN_ELV,X,Y\n")
    print("\tPoint Count GT "+str(Range))
    for i in data1:
        print("\t\tUNIQUEID : "+str(i[0]))
        i2 = RowAdd(i[1], Range)
        for j in range(len(i2)):
            fw = i2[j:j+(2*Range)+1]
            if len(fw) == ((2*Range)+1):
                mdlist = [float(k[1]) for k in fw]
                mdsort = sorted(mdlist)
                median = mdsort[Range]
                temp3 = str(i[0])+","+str(i[0])+"_"+str(fw[Range][0])+","+str(fw[Range][1])+","+str(median)+","+str(fw[Range][2])+","+str(fw[Range][3])+"\n"
                w.writelines(temp3)
                
    print("\tPoint Count LT "+str(Range))
    for ii in data2:
        for jj in ii[1]:
            temp4 = str(ii[0])+","+str(ii[0])+"_"+str(jj[0])+","+str(jj[1])+","+str(jj[1])+","+str(jj[2])+","+str(jj[3])+"\n"
            w.writelines(temp4)
            
    w.close()
    fopen.close()

    print("Final Median Table : "+str(WorkingFolder+"\\Table_"+str(os.path.basename(OutRiverPoints1))+"_Median"))
    arcpy.TableToTable_conversion(outfile, WorkingFolder, "Table_"+str(os.path.basename(OutRiverPoints1))+"_Median")

    etime2 = time.time()
    tdiff2 = etime2 - stime2
    timeit("", tdiff2)
    print("Step-2 Done!")

# PHASE-3

"""
Following are required fields:
-----------------------------
UNIQUEID
XSCL_ID
MEDIAN_ELV
"""

# xlist contains "XSCL_ID" and "MEDIAN_ELV" fields
def JumpSolve(xlist):
    # Reversed the list from F-Node to T-Node
    ylist = list()
    for i in reversed(xlist):
        ylist.append(i)
        del i
    del xlist

    print("\tTotal Points : "+str(len(ylist)))

    # Extract the sudden jump point id(s)
    delist = list()
    for i in range(len(ylist)):
        cdata = ylist[i+1:]
        for j in cdata:
            if ylist[i][1]+dz < j[1]:
                delist.append(j)
            del j
        del i

    definal = list(set(delist))

    f_id = int(ylist[0][0].split("_")[1])
    t_id = int(ylist[-1][0].split("_")[1])

    # Sort the sudden jump point id(s)
    desort = list()
    for i in range(t_id, f_id+1):
        for j in definal:
            if ylist[0][0].split("_")[0]+"_"+str(i) == j[0]:
                desort.append(j)
            del j
        del i
    del definal

    # Use filter to get the closest point id values for error point id
    alist = list(set(ylist)-set(desort))
    flist = list(set(alist))
    for j in range(len(desort)):
        print("\t\tIteration Number : "+str(j+1))
        nvlist = list()
        nv = None
        for i in desort:
            # String search from list
            f_str = i[0].split("_")[0]+"_"+str(int(i[0].split("_")[1])-1)
            f_val = filter(lambda x: f_str in x, flist)

            p_str = i[0].split("_")[0]+"_"+str(int(i[0].split("_")[1])+1)
            p_val = filter(lambda y: p_str in y, flist)

            if f_val == [] and p_val:
                nv = (i[0], p_val[0][1])
                desort = list(set(desort) - set([i]))
            elif f_val and p_val == []:
                nv = (i[0], f_val[0][1])
                desort = list(set(desort) - set([i]))
            elif f_val and p_val:
                nv = (i[0], f_val[0][1])
                desort = list(set(desort) - set([i]))
            else:
                pass

            if nv:
                nvlist.append(nv)

            del i
        if nvlist:
            sortnv = list(set(nvlist))
            for k in sortnv:
                flist.append(k)
                del k
        else:
            break
        del j
    del desort
    del alist

    # Final List
    fn_list = list()
    for i in range(t_id, f_id+1):
        for j in flist:
            if flist[0][0].split("_")[0]+"_"+str(i) == j[0]:
                fn_list.append(j)
            del j
        del i
    del flist

    fn_list2 = list(set(fn_list))

    print("\tOutput Points : "+str(len(fn_list2)))

    if len(ylist) == len(fn_list2):
        print("\tPoint Count is Match!")
    else:
        print("\tCheck Data!")

    return fn_list2

if GetFilteredRiverElevation == 3:
    stime3 = time.time()

    if os.path.exists(outtxt):
        os.remove(outtxt)
        outtxt
    else:
        outtxt

    # Call the above function
    # Get all the required fields
    rows = open(outfile, 'r')

    # Read the point featureclass
    xtemp = list()
    for row in rows:
        rstrip = row.strip()
        rsplit = rstrip.split(",")
        xtemp.append(rsplit)
        del row
    del rows
    xtemp1 = xtemp[1:]
    xdisc = {}
    for rw in xtemp1:
        #print rw
        if not rw[0] in xdisc:
            xdisc[rw[0]] = list()
        xdisc[rw[0]].append((rw[1], float(rw[3])))
        del rw
    del xtemp
    del xtemp1

    # Write into text file
    w = open(outtxt, 'w')
    w.writelines("XSCLID,NEWVAL\n")
    for k in xdisc.keys():
        print("\tUNIQUEID : "+str(k))
        xdlist = xdisc[k]
        fnOutput = JumpSolve(xdlist)
        for i in fnOutput:
            w.writelines(",".join(str(j) for j in i)+"\n")
            del i
        del k
    w.close()
    
    # Export text file
    arcpy.TableToTable_conversion(outtxt, WorkingFolder, "Table_"+str(os.path.basename(OutRiverPoints1))+"_MedianFilter")

    # Del files
    if os.path.exists(outfile):
        os.remove(outfile)
    if os.path.exists(outtxt):
        os.remove(outtxt)

    etime3 = time.time()
    tdiff3 = etime3 - stime3
    timeit("", tdiff3)
    print("Step-3 Done!")
        
# Del files
if os.path.exists(os.path.dirname(WorkingFolder)+"\\Table_"+str(os.path.basename(OutRiverPoints1))+".txt"):
    os.remove(os.path.dirname(WorkingFolder)+"\\Table_"+str(os.path.basename(OutRiverPoints1))+".txt")
if os.path.exists(os.path.dirname(WorkingFolder)+"\\Table_"+str(os.path.basename(OutRiverPoints1))+".txt.xml"):
    os.remove(os.path.dirname(WorkingFolder)+"\\Table_"+str(os.path.basename(OutRiverPoints1))+".txt.xml")

# Join Tables
print ("Join Tables")
flayer = WorkingFolder+"\\"+"f_Lyr"
arcpy.MakeFeatureLayer_management(OutRiverPoints1,flayer)
arcpy.AddJoin_management(flayer, "XSCL_UID", WorkingFolder+"\\Table_"+str(os.path.basename(OutRiverPoints1))+"_MedianFilter", "XSCLID", "KEEP_ALL")
arcpy.CopyFeatures_management(flayer, OutRiverPoints)

if arcpy.Exists(OutRiverPoints1):
    arcpy.Delete_management(OutRiverPoints1, "")
if arcpy.Exists(WorkingFolder+"\\Table_"+str(os.path.basename(OutRiverPoints1))+"_MedianFilter"):
    arcpy.Delete_management(WorkingFolder+"\\Table_"+str(os.path.basename(OutRiverPoints1))+"_MedianFilter", "")

# Mosaic DTM and River Raster
print("Get the Hydroburn DTM!")
PointToRaster = WorkingFolder+"\\Pnt2Rast"
if arcpy.Exists(PointToRaster):
    arcpy.Delete_management(PointToRaster, "")
f_elevation = "Table_"+str(os.path.basename(OutRiverPoints1))+"_MedianFilter"+"_NEWVAL"
desc = arcpy.Describe(InDem)
cellsize = desc.children[0].meanCellHeight
tempEnvironment0 = arcpy.env.snapRaster
arcpy.env.snapRaster = InDem
arcpy.PointToRaster_conversion(OutRiverPoints, str(f_elevation), PointToRaster, "MOST_FREQUENT", "NONE", str(cellsize))
arcpy.env.snapRaster = tempEnvironment0

# Process: Mosaic To New Raster
tempEnvironment0 = arcpy.env.snapRaster
arcpy.env.snapRaster = InDem
tempEnvironment1 = arcpy.env.extent
arcpy.env.extent = desc.extent
arcpy.MosaicToNewRaster_management(str(InDem)+";"+str(PointToRaster), WorkingFolder, str(os.path.basename(InDem))+"_Hydroburn", "", "32_BIT_FLOAT", "", "1", "MINIMUM", "FIRST")
arcpy.env.snapRaster = tempEnvironment0
arcpy.env.extent = tempEnvironment1

if arcpy.Exists(PointToRaster):
    arcpy.Delete_management(PointToRaster, "")
if arcpy.Exists(OutRiverPoints):
    arcpy.Delete_management(OutRiverPoints, "")
    
etime4 = time.time()
tdiff4 = etime4 - stime
timeit("Total time is ", tdiff4)
print("All Done!")
