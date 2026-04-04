# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 2016

@author: Debabrata Ghorai, Ph.D.

Identify sink and flat in DEM and then assign sequence from 1 to n.
"""

import numpy


def find_sinknflat_dem(dem_rast):
    """Function for Sink and Flat Identification in DEM"""
    # D8 diagonal and horizontal distances
    d, h, x = numpy.sqrt(2), 1.0, 0.0
    d8dist = numpy.array([[d, h, d], [h, x, h], [d, h, d]])

    # identify sink and flat pixels and replace with 1
    dem_array = numpy.copy(dem_rast)
    sf_array = numpy.zeros(dem_array.shape, dtype=int)
    for i in range(dem_array.shape[0]):
        for j in range(dem_array.shape[1]):
            d8 = dem_array[i:i+3, j:j+3]
            if d8.shape == (3, 3):
                cslope = (d8 - d8[1, 1])*d8dist
                melv = d8[numpy.where(cslope == cslope.min())]
                if len(melv) == 1:
                    # identify sink
                    if d8[1, 1] == melv[0]:
                        sf_array[i+1, j+1] = 1
                elif len(melv) > 1:
                    # identify flat
                    if d8[1, 1] == melv[0]:
                        sf_array[i+1, j+1] = 1
                else:
                    pass
            del j
        del i

    # assign unique number to all pixel value 1 row-wise
    unique_row = numpy.zeros(sf_array.shape, dtype=int)
    for i in range(sf_array.shape[0]):
        cnt = 1
        for j in range(sf_array.shape[1]):
            if sf_array[i, j] > 0:
                unique_row[i, j] = cnt
                cnt += 1
            del j
        del i

    # assign unique values to all unique row column-wise
    n = 0
    unique_col = numpy.zeros(unique_row.shape, dtype=int)
    for i in range(unique_row.shape[0]):
        if max(unique_row[i]) > 0:
            for j in range(unique_row.shape[1]):
                if unique_row[i, j] > 0:
                    unique_col[i, j] = unique_row[i, j]+n
                del j
            n = max(unique_row[i])+n
        del i

    return unique_col


if __name__ == '__main__':
    # input dem
    DEM = numpy.array([[11, 13, 12, 12, 12, 11, 12, 12, 12, 11, 11, 19],
                       [10, 12, 11, 11, 11, 11, 12, 11, 11, 12, 12, 20],
                       [12, 13, 11, 10, 12, 12, 12, 11, 11, 11, 12, 22],
                       [13, 12, 11, 11, 11, 11, 10, 11, 11, 10, 11, 20],
                       [12, 10, 10, 9, 10, 11, 10, 12, 11, 11, 10, 21],
                       [11, 10, 9, 9, 10, 11, 11, 11, 8, 10, 10, 21],
                       [11, 11, 9, 10, 11, 11, 11, 9, 7, 9, 10, 19],
                       [13, 10, 8, 8, 10, 13, 10, 12, 11, 10, 10, 23],
                       [12, 8, 8, 4, 2, 2, 7, 10, 12, 12, 13, 20],
                       [9, 11, 8, 9, 8, 0, 6, 11, 10, 10, 9, 20],
                       [9, 4, 11, 7, 10, 9, 7, 8, 7, 7, 6, 15],
                       [8, 9, 12, 10, 11, 12, 14, 9, 8, 4, 3, 10]])
    # call the function
    sfs = find_sinknflat_dem(DEM)
