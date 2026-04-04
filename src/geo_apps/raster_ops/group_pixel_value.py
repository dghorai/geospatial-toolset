# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 2016

@author: Debabrata Ghorai, Ph.D.

Group raster pixel values.
"""

import numpy


def group_raster_pixels(data):
    """
    Raster pixel clustering.
    """
    # pixel to pixel connection list
    ftnode = list()
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # create 3x3 matrix
            m = data[i:i+3, j:j+3]
            if m.shape == (3, 3):
                # query: if center of 3x3 matrix > 0
                if m[1, 1] > 0:
                    # matrix to list conversion
                    m2 = m.tolist()
                    m3 = m2[0]+m2[1]+m2[2]
                    # remove center pixel value from list
                    m3.remove(m[1, 1])
                    # query: if pixel value > 0 then append to another list
                    m4 = list()
                    for k in m3:
                        if k > 0:
                            m4.append(k)
                        del k
                    # also append center pixel value to that another list
                    ftnode.append(([m[1, 1]]+m4))
            del j
        del i

    # return ftnode
    len_l = len(ftnode)
    ii = 0
    while ii < (len_l-1):
        for jj in range(ii+1, len_l):
            # i,j iterate over all pairs of ftnode's elements including new
            # elements from merged pairs. We use len_l because len(l)
            # may change as we iterate
            i_set = set(ftnode[ii])
            j_set = set(ftnode[jj])
            if len(i_set.intersection(j_set)) > 0:
                # remove these two from list
                ftnode.pop(jj)
                ftnode.pop(ii)
                # merge them and append to the orig. list
                ij_union = list(i_set.union(j_set))
                ftnode.append(ij_union)
                # len(l) has changed
                len_l -= 1
                # adjust 'i' because elements shifted
                ii -= 1
                # abort inner loop, continue with next l[i]
                break
        ii += 1
    return ftnode


if __name__ == "__main__":
    raster_data = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 2, 0, 0, 0],
                               [0, 0, 3, 0, 0, 0, 4, 0],
                               [0, 5, 0, 0, 6, 7, 8, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0]])
    # call the function
    rcluster = group_raster_pixels(raster_data)
