# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 21:23:01 2022

@author: Debabrata Ghorai, Ph.D.
"""

import os

pic_dir = r"D:\Assignment\data"

file_list = os.listdir(pic_dir)

for i, f in enumerate(file_list):
    if f.endswith(".jpg"):
        dst = f"smartphone_{str(i+1)}.jpg"
        src = f"{pic_dir}/{f}"
        dst =f"{pic_dir}/{dst}"
        # rename file
        os.rename(src, dst)
