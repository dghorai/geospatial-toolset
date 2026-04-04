#!/usr/bin/env python
"""************************************************************
* Created:        January 9, 2020
* Author:         Debabrata Ghorai
* Purpose:        Get all files into a list from a directory
************************************************************"""
# Import Modules
import os

# Input Arguments
root = r"E:"
outfile = r"C:\file_list.txt"

# Get All Files
w = open(outfile, 'w')
for path, subdirs, files in os.walk(root):
    for name in files:
        w.writelines(os.path.join(path, name)+"\n")
w.close()
print ("Done!")
