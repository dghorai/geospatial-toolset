#! C:\Python27\ArcGISx6410.5\python.exe
#-------------------------------------------------------------------------
# Convert DXF CAD File to Geodatabase File
# Author: Debabrata Ghorai
# Created on: 2018-11-30 (YYYY-MM-DD)
# Usage: Italy flood modeling project
# ------------------------------------------------------------------------
import os
import arcpy

# Input Arguments
InputDxfFolder = r"D:\Lazio_Points\Part3\DXF"
OutputGDB = r"D:\Lazio_Points\Part3\Part3_2.gdb"


# File Processing
def DXF_to_FC(dxfFolder, outGDB):
    fileList = os.listdir(dxfFolder)
    for f in fileList:
        print (f)
        cad_file_path = dxfFolder+"\\"+f
        feat_dataset_name = "f"+str(f.split(".")[0])
        arcpy.CADToGeodatabase_conversion(cad_file_path, outGDB, feat_dataset_name, "1000")
        del f
    return

if __name__ == "__main__":
    DXF_to_FC(InputDxfFolder, OutputGDB)
    print ("Done!")
