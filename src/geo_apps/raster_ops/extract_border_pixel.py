# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 2016

@author: Debabrata Ghorai, Ph.D.

Border pixel value extraction from raster data.
"""

import numpy


def get_border_pixel_values(dem):
    """Extract border pixel values"""
    # edge detection for row
    r_edge = list()
    for i in range(dem.shape[0]):
        r_list = list()
        for j in range(dem.shape[1]):
            if dem[i, j] != 0:
                r_list.append(dem[i, j])
            del j
        r_edge.append([r_list[0], r_list[-1]])
        del r_list
        del i
    r_edge2 = sum(r_edge, [])
    del r_edge
    # transpose rows to columns
    t_dem = dem.T
    # edge detection for column
    c_edge = list()
    for i in range(t_dem.shape[0]):
        c_list = list()
        for j in range(t_dem.shape[1]):
            if t_dem[i, j] != 0:
                c_list.append(t_dem[i, j])
            del j
        c_edge.append([c_list[0], c_list[-1]])
        del c_list
        del i
    c_edge2 = sum(c_edge, [])
    del c_edge
    del t_dem
    # concatenation two vector/list
    f_edge = list(set(r_edge2+c_edge2))
    f_edge.sort()
    return f_edge


if __name__ == "__main__":
    raster_data = numpy.array([[0, 0, 12, 21, 15, 20, 0, 0],
                               [1, 13, 4, 5, 6, 0, 0, 0],
                               [0, 11, 14, 10, 16, 18, 0, 0],
                               [22, 32, 0, 0, 0, 7, 81, 9]])
    pixeledge = get_border_pixel_values(raster_data)
    print(pixeledge)
