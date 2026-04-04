'''import xlwt
#Create workbook and worksheet
wkbook = xlwt.Workbook()
wksheet = wkbook.add_sheet('FAO_Soil_Legend')
row = 0 #Row Counter
fileOpen = open('E:/Text/New Text Document.txt')
for line in fileOpen:
    #Seperate fields by commas
    L = line.strip()
    wksheet.write(row,0,L)
    row = row+1
    print L

wkbook.save('E:/Text/FAOSoil.xls')
'''

import os

infile = open('E:/Text/New Text Document.txt','r')

readfile = infile.read()
infile.close()

outfile = open('E:/Text/Txt.xls','w')
outfile.writelines(readfile)
outfile.flush()
outfile.close()
