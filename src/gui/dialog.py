# -*- coding: utf-8 -*-
"""
Created on Tue May 12 19:10:11 2026

@author: Debabrata Ghorai, Ph.D.

GUI common functions.

"""

import tkinter as tk
import streamlit as st

from tkinter import filedialog


# SET new shapefile path in GUI
def set_new_shapefile_path():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True) # Forces the window to the front
    # Open the Save As dialog
    # We specify .shp as the default extension
    file_path = filedialog.asksaveasfilename(
        defaultextension=".shp",
        filetypes=[("Shapefile", "*.shp"), ("All Files", "*.*")],
        title="Save Shapefile As"
    )
    root.destroy()
    # Check if the user cancelled the dialog
    if not file_path:
        st.write("Save operation cancelled.")
        return
    return file_path