# xlsx to xls conversion in python

# import module
import win32com.client
xl = win32com.client.Dispatch("Excel.Application")

# Input argument
infile = r"G:\Data\Crop data.xlsx"
outxls = r"G:\XLS\CropData.xls"

# Conversion
wb = xl.Workbooks.Open(infile)
wb.SaveAs(outxls, FileFormat = 56)
wb.Close()
xl.Quit()
