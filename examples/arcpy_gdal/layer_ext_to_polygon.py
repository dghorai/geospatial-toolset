import os
import arcpy

from arcpy import env

# Input arguments
inFC = r"C:\Work\Inputs.gdb\Dissolve_BND"

def Extent_to_Polygon(infc, extentPoly):
    # Feature extent
    extent = arcpy.Describe(infc).extent

    # Array to hold points
    array = arcpy.Array()

    # Create the bounding box
    array.add(extent.lowerLeft)
    array.add(extent.lowerRight)
    array.add(extent.upperRight)
    array.add(extent.upperLeft)

    # ensure the polygon is closed
    array.add(extent.lowerLeft)

    # Create the polygon object
    polygon = arcpy.Polygon(array)
    array.removeAll()

    # save to disk
    arcpy.CopyFeatures_management(polygon, extentPoly)
    del polygon

def main():
    outpath = os.path.dirname(inFC)
    shpLst = ["Polygon","Polyline","Point","MultiPoint","MultiPatch"]
    try:
        sType = arcpy.Describe(inFC).shapeType
        for s in shpLst:
            if sType == s:
                print("File does exist.")
                if arcpy.Describe(inFC).extension == "":
                    outfile = outpath+"\\"+inFC.split("\\")[-1]+"_polygon"
                else:
                    outfile = outpath+"\\"+inFC.split("\\")[-1].split(".")[0]+"_polygon.shp"
                Extent_to_Polygon(inFC, outfile)
    except:
        print("File does not exit.")
        
if __name__ == '__main__':
    main()