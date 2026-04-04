# PDF TO IMAGE CONVERSION
#========================
# Scripted by Debabrata Ghorai (ghoraideb@gmail.com)
# Task: PDF to Image Conversion
# Process: i) Multipage PDF to Singlepage PDF, ii) PDF to Image

# Import Modules
from pyPdf import PdfFileReader, PdfFileWriter
import PythonMagick, os

# Input arguments
PdfFolder = r"E:\Bhuvan_LULC_ReferenceLayer\PDF"
OutFolder = r"E:\Bhuvan_LULC_ReferenceLayer\JPG"

# Local variables
BatchPdfFiles = os.listdir(PdfFolder)

# Process: Multipage PDF to Singlepage PDF
for bpf in BatchPdfFiles:
    fileext = bpf.split(".")[-1]
    if fileext == "pdf":
        print(bpf)
        PdfFile = PdfFolder+"\\"+bpf
        OpenPdfFile = open(PdfFile, 'rb')
        ReadPdfFile = PdfFileReader(OpenPdfFile)
        PdfPages = ReadPdfFile.getNumPages()
        PdfFileName = PdfFile.split("\\")[-1].split(".")[0]

        for p in range(PdfPages):
            pw = PdfFileWriter()
            pw.addPage(ReadPdfFile.getPage(p))
            name = str(PdfFileName)+"_"+str(p+1)
            print(name)
            w = open(OutFolder+"\\"+"%s.pdf" % name, 'wb')
            pw.write(w)
            w.close()
    else:
        pass

# Process: PDF to Image
pdfs = os.listdir(OutFolder)
for i in pdfs:
    iname = i.split(".")[0]+".tif"
    ipdf = OutFolder+"\\"+i
    img = PythonMagick.Image()
    img.density('300')
    img.read(ipdf) # read in at 300 dpi
    bgcolor = PythonMagick.Color('#ffffff') # White background
    size = "%sx%s" % (img.columns(), img.rows())
    newimg = PythonMagick.Image(size, bgcolor)
    newimg.type = img.type
    newimg.composite(img, 0, 0, PythonMagick.CompositeOperator.SrcOverCompositeOp)
    newimg.write(OutFolder+"\\"+iname)
    os.remove(ipdf)
    print(iname)

print("PDF to Image Conversion Done!")
