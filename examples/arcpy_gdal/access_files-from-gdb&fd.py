# List all fc from gdb and feature dataset
# Debabrata Ghorai
# Date: May 12, 2020
#-------------------------------------------------------

# Import Modules
import os
import time
import arcpy
import datetime


# Time function
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
    return

# fc in feature dataset
def fc_in_fd(workspace, fcname, fctype, outgdb, n):
    arcpy.env.workspace = workspace
    datasetList = arcpy.ListDatasets('*','Feature')
    for dataset in datasetList:
        arcpy.env.workspace = dataset
        fcList = arcpy.ListFeatureClasses()
        for fc in fcList:
            datapath = arcpy.env.workspace+"\\"+fc
            if fc == fcname:
                fcdesc = arcpy.Describe(datapath)
                shptype = fcdesc.shapeType
                #print (shptype)
                if shptype == fctype:
                    print ("\t"+str(fc))
                    arcpy.CopyFeatures_management(datapath, outgdb+"\\"+fcdesc.basename+str(n))
                else:
                    pass
            else:
                pass
    return

# fc in file geodatabase
def fc_in_gdb(workspace, fcname, fctype, outgdb, n):
    arcpy.env.workspace = workspace
    fcList = arcpy.ListFeatureClasses()
    for fc in fcList:
        datapath = arcpy.env.workspace+"\\"+fc
        if fc == fcname:
            fcdesc = arcpy.Describe(datapath)
            shptype = fcdesc.shapeType
            if shptype == fctype:
                print ("\t"+str(fc))
                arcpy.CopyFeatures_management(datapath, outgdb+"\\"+fcdesc.basename+str(n))
            else:
                pass
        else:
            pass
    return

# Get list of file geodatabase
def get_gdb_list(root):
    gdbList = list()
    for path, subdirs, files in os.walk(root):
        for name in subdirs:
            dirname = os.path.join(path, name)
            if dirname.split("\\")[-1].endswith(".gdb"):
                gdbList.append(dirname)
            else:
                pass
    return gdbList

# main function
def main():
    folder = r"\\edc\NewData_2020\Part01"
    out_gdb = r"E:\MFL_Merging\Part01.gdb"
    fc_name = 'NHN_HN_PrimaryDirectedNLFlow_1'
    fc_type = "Polyline"

    t0 = time.time()
    tot_gdb = get_gdb_list(folder)
    n = 1
    cnt = 1
    for gdb in tot_gdb:
        print (cnt)
        print (gdb)
        fc_in_fd(gdb, fc_name, fc_type, out_gdb, n)
        n += 1
        fc_in_gdb(gdb, fc_name, fc_type, out_gdb, n)
        n += 1
        cnt += 1
    timeit("Total Time: ", time.time() - t0)
    return


if __name__ == '__main__':
    main()
