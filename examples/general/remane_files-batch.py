# Import Module
import os
import shutil

# Rename tsunami simulation files: example, tsu_time_00480.grd

fnum = 10000
intv = 20
inDir = r"F:\Output\Tsunami_Simulation\New folder"
inFileList = os.listdir(inDir)

for i, j in enumerate(inFileList):
    fn = j.split("_")[-1].split(".")[0]
    num = fnum+intv*(i+1)
    if len(str(num)) == 1:
        nm = '0000'+str(num)
    elif len(str(num)) == 2:
        nm = '000'+str(num)
    elif len(str(num)) == 3:
        nm = '00'+str(num)
    elif len(str(num)) == 4:
        nm = '0'+str(num)
    else:
        nm = str(num)
    newName = 'tsu_time_'+str(nm)+'.grd'
    os.rename(inDir+"\\"+j, inDir+"\\"+newName)
