# Author: Debabrata Ghorai
# Task: Image to text conversion
# Modules: PyTesser, PIL

from pytesser import *
from PIL import Image

# Input variables
img = Image.open(r"E:\PhD_Working\Published_Maps\salinity_map.jpg")
outfile = open(r"E:\PhD_Working\Work\writer1.txt", 'w')

print ("Image to String Conversion Started...")
# Local variables
ImgToText = image_to_string(img)
Text = ImgToText.strip()
Strings = Text.split('\n')

# Processing
cnt = 1
for row in Strings:
    w = row+"\n"
    outfile.writelines(w)
    print ("Line number: %s" % cnt)
    cnt += 1

print ("Image to String Conversion Done!")
