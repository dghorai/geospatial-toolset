import sys
import arcpy
import src.script_dem as dem
import src.script_text as txt

# config env
arcpy.env.overwriteOutput = True
arcpy.CheckOutExtension("Spatial")

input1 = arcpy.GetParameterAsText(0)
input2 = arcpy.GetParameterAsText(1)
#--------------------------------------
workfolder = arcpy.GetParameterAsText(2)
intxtfile = arcpy.GetParameterAsText(3)
inxscl = arcpy.GetParameterAsText(4)
indem = arcpy.GetParameterAsText(5)
#------------------------------------
outputfile = arcpy.GetParameterAsText(6)

if str(input1) == "true":
    txt.Min_From_Textfile(workfolder, intxtfile, inxscl, outputfile)
elif str(input2) == "true":
    dem.XSCL_QAQC_Layer_I(workfolder, inxscl, indem, outputfile)
else:
    arcpy.AddMessage("Select Checkbox!")
    sys.exit()
