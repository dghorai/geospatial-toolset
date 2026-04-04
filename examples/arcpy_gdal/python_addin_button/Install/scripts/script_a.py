# Import modules
import sys
import arcpy
import src.merge as merge
import src.qaqc_flags_a as xa
import src.qaqc_flags_b as xb
import src.qaqc_flags_c as xc
import src.qaqc_flags_d as xd
import src.qaqc_flags_e as xe
import src.qaqc_flags_f as xf
import src.qaqc_flags_g as xg
import src.qaqc_flags_h as xh

# config env
arcpy.env.overwriteOutput = True
arcpy.CheckOutExtension("Spatial")

# Input arguments
#-------------------------------------
input_a = arcpy.GetParameterAsText(0) # Broken MFL
input_b = arcpy.GetParameterAsText(1) # Shorter MFL
input_c = arcpy.GetParameterAsText(2) # XS WO MFL
input_d = arcpy.GetParameterAsText(3) # MFL WO XS
input_e = arcpy.GetParameterAsText(4) # Wider, Closer and Multi-X XS between XS and MFL
input_f = arcpy.GetParameterAsText(5) # Shorter, Near, Outside, and Croses among XS, MFL, and MB
input_g = arcpy.GetParameterAsText(6) # Multi-X XS among XS
input_h = arcpy.GetParameterAsText(7) # XS with Null Elevation
#----------------------------------------
demresolution = float(arcpy.GetParameterAsText(8)) #--------add-------
workfolder = arcpy.GetParameterAsText(9)
modelflowline = arcpy.GetParameterAsText(10)
crosssection = arcpy.GetParameterAsText(11)
modelboundary = arcpy.GetParameterAsText(12)
digitalelvmodel = arcpy.GetParameterAsText(13)
#-------------------------------------------------
minrivlength = float(arcpy.GetParameterAsText(14))
widerdistance = float(arcpy.GetParameterAsText(15))
closerdistance = float(arcpy.GetParameterAsText(16))
#----------------------------------------------
qaqcoutput = arcpy.GetParameterAsText(17)
#-------------------------------------------------
minxslength = demresolution*5.0 # minimum xs length is 5 times of DTM resolution
minxslenghtho = demresolution*12.0 # minimum xs length of 3rd and above is 12 times of DTM resolution

# Function calling statements
if (str(input_a) == "true") or (str(input_b) == "true") or (str(input_c) == "true") or (str(input_d) == "true") or (str(input_e) == "true") or (str(input_f) == "true") or (str(input_g) == "true") or (str(input_h) == "true"):
    if str(input_a) == "true":
        xa.XSCL_QAQC_Layer_A(modelflowline, workfolder)
    if str(input_b) == "true":
        xb.XSCL_QAQC_Layer_B(modelflowline, workfolder, minrivlength)
    if str(input_c) == "true":
        xc.XSCL_QAQC_Layer_C(crosssection, modelflowline, workfolder)
    if str(input_d) == "true":
        xd.XSCL_QAQC_Layer_D(modelflowline, crosssection, workfolder)
    if str(input_e) == "true":
        xe.XSCL_QAQC_Layer_E(crosssection, modelflowline, workfolder, widerdistance, closerdistance)
    if str(input_f) == "true":
        xf.XSCL_QAQC_Layer_F(workfolder, crosssection, modelflowline, modelboundary, minxslength, minxslenghtho)
    if str(input_g) == "true":
        xg.XSCL_QAQC_Layer_G(workfolder, crosssection)
    if str(input_h) == "true":
        xh.XSCL_QAQC_Layer_H(workfolder, crosssection, digitalelvmodel)
else:
    arcpy.AddMessage("Select Checkbox!")
    sys.exit()

# List of comments
cm1 = 'Split MFL'
cm2 = 'Shorter MFL'
cm3 = 'XS WO MFL'
cm4 = 'MFL WO XS'
cm5 = 'No XS WI '+str(widerdistance)+'m'
cm6 = 'Too Close XS'
cm7 = 'Multi-X MFLnXS'
cm8 = 'Shorter XS on All MFL'
cm9 = 'Shorter XS on 3rd & Above'
cm10 = 'XS Near JUNC'
cm11 = 'XS Out of BND'
cm12 = 'MFL Out of BND'
cm13 = 'XS Cross BND'
cm14 = 'MFL Cross BND'
cm15 = 'Multi-X XS'
cm16 = 'XS on NoData'

# List of output files
out1 = workfolder+"\\"+"broken_mfl_output"
out2 = workfolder+"\\"+"mrl_mfl_output"
out3 = workfolder+"\\"+"xscl_wo_mfl"
out4 = workfolder+"\\"+"mfl_wo_xscl"
out5 = workfolder+"\\"+"t_wider_mfl"
out6 = workfolder+"\\"+"t_closer_mfl"
out7 = workfolder+"\\"+"t_output_xs"
out8 = workfolder+"\\"+"t_model1"
out9 = workfolder+"\\"+"t_xscl2"
out10 = workfolder+"\\"+"t_model3"
out11 = workfolder+"\\"+"t_model4"
out12 = workfolder+"\\"+"t_model5"
out13 = workfolder+"\\"+"t_model6"
out14 = workfolder+"\\"+"t_model7"
out15 = workfolder+"\\"+"t_model_output15"
out16 = workfolder+"\\"+"t_model16"
#------------------------------------------------------
cmlist = [cm1, cm2, cm3, cm4, cm5, cm6, cm7, cm8, cm9, cm10, cm11, cm12, cm13, cm14, cm15, cm16]
fclist = [out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, out16]

# Check for existance of output files
fclist_final = list()
cmlist_final = list()
for i, j in zip(fclist, cmlist):
    try:
        if (arcpy.Exists(i) == True):
            count = str(arcpy.GetCount_management(i))
            if count != "0":
                fclist_final.append(i)
                cmlist_final.append(j)
            else:
                arcpy.Delete_management(i, "")
    except Exception as e:
        arcpy.AddMessage(e.message)
    del i
    del j
del cmlist
del fclist

# Merge all available feature class
merge.Update_Cursor(fclist_final, cmlist_final, qaqcoutput)

# Delete file if exixts
for ii in fclist_final:
    if arcpy.Exists(ii):
        arcpy.Delete_management(ii, "")
    del ii

# Check all the field in the output file
mfield = arcpy.ListFields(qaqcoutput)
alflds = list()
for ff in mfield:
    if ff.name == "Comments":
        pass
    elif (ff.name == "Shape_Length") or (ff.name == "Shape_Leng"):
        pass
    elif ff.name == "Shape":
        pass
    elif (ff.name == "OBJECTID") or (ff.name == "OBJECTID_1") or (ff.name == "OBJECTID_2"):
        pass
    else:
        alflds.append(ff.name)
arcpy.DeleteField_management(qaqcoutput, alflds)
