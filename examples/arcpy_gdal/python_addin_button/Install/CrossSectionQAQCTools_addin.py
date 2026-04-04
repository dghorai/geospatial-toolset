import arcpy
import pythonaddins
import os
relPath = os.path.dirname(__file__)
tpath = relPath + r"\XSQCToolBox.tbx"

class XSCLQAQCButtonA(object):
    """Implementation for CrossSectionQAQCTools_addin.button1 (Button)"""
    def __init__(self):
        self.enabled = True
        self.checked = False
    def onClick(self):
        pythonaddins.GPToolDialog(tpath, "XSCLQAQCLayerA")

class XSCLQAQCButtonB(object):
    """Implementation for CrossSectionQAQCTools_addin.button2 (Button)"""
    def __init__(self):
        self.enabled = True
        self.checked = False
    def onClick(self):
        pythonaddins.GPToolDialog(tpath, "XSCLQAQCLayerB")

class XSCLQAQCButtonC(object):
    """Implementation for CrossSectionQAQCTools_addin.button3 (Button)"""
    def __init__(self):
        self.enabled = True
        self.checked = False
    def onClick(self):
        pythonaddins.GPToolDialog(tpath, "XSCLQAQCLayerC")

class XSCLQAQCButtonD(object):
    """Implementation for CrossSectionQAQCTools_addin.button4 (Button)"""
    def __init__(self):
        self.enabled = True
        self.checked = False
    def onClick(self):
        pythonaddins.GPToolDialog(tpath, "XSCLQAQCLayerD")

class XSCLQAQCButtonE(object):
    """Implementation for CrossSectionQAQCTools_addin.button4 (Button)"""
    def __init__(self):
        self.enabled = True
        self.checked = False
    def onClick(self):
        pythonaddins.GPToolDialog(tpath, "XSCLQAQCLayerE")
