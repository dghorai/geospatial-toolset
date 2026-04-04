# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 19:48:13 2022

@author: Debabrata Ghorai, Ph.D.

"""

import wand
from wand.image import Image
import os

SourceFolder="K:/HeicFolder"
TargetFolder="K:/JpgFolder"

for file in os.listdir(SourceFolder):
   SourceFile=SourceFolder + "/" + file
   TargetFile=TargetFolder + "/" + file.replace(".HEIC",".JPG")
 
   img=Image(filename=SourceFile)
   img.format='jpg'
   img.save(filename=TargetFile)
   img.close()
