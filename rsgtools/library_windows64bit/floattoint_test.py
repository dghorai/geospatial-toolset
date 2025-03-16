# Author: Debabrata Ghorai, Ph.D.
#-------------------------------
# Import Modules
import os
import ctypes
import numpy

lib = ctypes.CDLL(os.getcwd()+'/library_windows64bit/floattoint.dll')
Multiply_and_Convert_Values = lib.Multiply_and_Convert_Values
Multiply_and_Convert_Values.restype = None
Multiply_and_Convert_Values.argtypes = [ctypes.c_int, ctypes.c_int, numpy.ctypeslib.ndpointer(ctypes.c_float), ctypes.c_float, numpy.ctypeslib.ndpointer(ctypes.c_int)]

arr = numpy.array([[1.0,2.0,3.0,4.0,5.0], [4.0,5.0,6.0,7.0,8.0], [7.0,8.0,9.0,1.0,2.0], [8.0,9.0,1.0,2.0,3.0], [9.0,1.0,2.0,3.0,4.0]], dtype=numpy.float32)
print("Original array")
print(arr)
out_arr = numpy.zeros(arr.shape, dtype=numpy.int32)
Multiply_and_Convert_Values(5, 5, arr, 1000, out_arr)
print("\n")
print("Modified array")
print(out_arr)
