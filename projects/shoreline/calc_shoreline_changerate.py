# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 18:41:31 2024

@author: Debabrata Ghorai, Ph.D.

SHORELINE CHANGE RATE ANALYSIS
------------------------------
Measurement of shoreline change rate using following statistical models:
1) Shoreline Change Envelope (SCE)
2) Net Shoreline Movement (NSM)
3) End Point Rate (EPR)
4) Linear Regression Rate-of-Change(LRR)
5) Weighted Linear Regression (WLR)

Notes:
1) Shoraline Change-Rate Computation - (Reference - DSAS)
2) Considering Leap Year Days (366) and Not a Leap Year Days (365)

Model references:
http://woodshole.er.usgs.gov/project-pages/DSAS/version4/images/pdf/DSASv4.pdf
http://www.soest.hawaii.edu/coasts/nps/erosionHazards.php
https://woodshole.er.usgs.gov/discus/messages/2/785.html?1404737665
http://answers.google.com/answers/threadview/id/761806.html

"""

# Import modules
import re
import numpy

from src.utils.geo_utils import leap_year
from logger import CustomException


class ShorelineChangerate:
    """
    Required Fields and Index:
        a) Transect ID (Index 0)
        b) Year ID (Index 1)
        c) Date (Index 2)
        d) Distance (Index 3)
        e) E (Index 4)

    Date Format: MM/DD/YYYY
    """

    def __init__(self):
        pass

    def get_data(self, infile):
        # Read transect data for the temporal shorelines
        fileopen = open(infile, 'r')
        datalist1 = list()
        for f in fileopen:
            dt = re.sub('\n', '', f)
            dt = re.split(',', dt)
            datalist1.append(dt)
        datalist2 = datalist1[1:]
        transect = list()
        for ts in datalist2:
            transect.append(int(ts[0]))
        unique = list(set(transect))
        return datalist2, unique

    def _sce(self, y_list):
        # ------------------------------
        # SCE - Shoreline Change Envelope
        # SCE = Greatest distance between all shorelines
        # -----------------------------------------------
        max_dist = max(y_list)
        diff_list = list()
        for dist in y_list:
            diff_list.append(max_dist - dist)
        SCE = max(diff_list)
        return SCE

    def old_and_yound_time(self, yr_list, yr_dist_list):
        # --------------------------------------------
        # NSM - Net Shoreline Movement
        # NSM = Distance between oldest and youngest shorelines
        # ------------------------------------------------------
        old_year = min(yr_list)
        young_year = max(yr_list)
        for yr in yr_dist_list:
            if old_year == yr[0]:
                old_d = yr[1]
                old_t = yr[2]
            if young_year == yr[0]:
                young_d = yr[1]
                young_t = yr[2]
        return old_d, old_t, young_d, young_t

    def _nsm(self, yr_list, yr_dist_list):
        # --------------------------------------------
        # NSM - Net Shoreline Movement
        # NSM = Distance between oldest and youngest shorelines
        # ------------------------------------------------------
        old_d, _, young_d, _ = self.old_and_yound_time(yr_list, yr_dist_list)
        NSM = (young_d - old_d)
        return NSM

    def _epr(self, yr_list, yr_dist_list, nsm):
        # -----------------------------------------
        # EPR - End Point Rate
        # EPR = Distance between two years/Time between oldest and most recent shoreline
        # -------------------------------------------------------------------------------
        _, old_t, _, young_t = self.old_and_yound_time(yr_list, yr_dist_list)
        EPR = (nsm/(young_t - old_t))
        return EPR

    def _lrr(self, yr_list, xy_list, x_list, y_list, xsqr_list, ysqr_list):
        # ----------------------------------------
        # LRR - Linear Regression Rate-of-Change
        # LRR = Plot of shoreline date (year) v/s distance from baseline
        # ---------------------------------------------------------------
        sum_x = sum(x_list)
        sum_y = sum(y_list)
        sum_xsqr = sum(xsqr_list)
        sum_ysqr = sum(ysqr_list)
        sum_xy = sum(xy_list)
        n = int(numpy.size(yr_list))
        A = float((n*sum_xy) - (sum_x*sum_y))
        B = float((n*sum_xsqr) - (sum_x**2))
        slope = A/B
        intercept = float(sum_y - (slope*sum_x))/n
        if numpy.sign(intercept) == -1.0:
            slp = slope
            intp = intercept
        else:
            slp = slope
            intp = intercept
        D = float((n*sum_ysqr) - (sum_y**2))
        r = float(A/(numpy.sqrt(B*D)))  # Correlation co-efficient
        R2 = r**2  # Coefficient of determination
        return slp, intp, R2

    def _wlr(self, w_list, wx_list, wxsqr_list, wy_list, wxy_list, wgt_list):
        # --------------------------------------------------
        # WLR - Weighted Linear Regression
        # WLR = Plot of shoreline date (year) v/s distance from baseline with weighted values
        # ------------------------------------------------------------------------------------
        I = sum(w_list)
        II = sum(wx_list)
        III = sum(wxsqr_list)
        IV = sum(wy_list)
        V = sum(wxy_list)
        # If m = the slope of the line and b = its intercept, then they are the solutions
        # to the following two equations:
        # I*b + II*m = IV
        # II*b + III*m = V
        E = I*III-II**2
        b = (III*IV - II*V)/E
        m = (I*V - II*IV)/E
        if numpy.sign(b) == -1.0:
            slp = m
            intp = b
        else:
            slp = m
            intp = b
        mean_x = II/I
        mean_y = IV/I
        sxx_list = list()
        syy_list = list()
        sxy_list = list()
        for j in wgt_list:
            sxx_list.append(j[2]*(j[0]-mean_x)**2)
            syy_list.append(j[2]*(j[1]-mean_y)**2)
            sxy_list.append(j[2]*(j[0]-mean_x)*(j[1]-mean_y))
        s_xx = sum(sxx_list)
        s_yy = sum(syy_list)
        s_xy = sum(sxy_list)
        wr = float(s_xy/(numpy.sqrt(s_xx*s_yy)))
        WR2 = wr**2
        return slp, intp, WR2

    def extract_shoreline_charngerate_params(self, datalist2, u, date_sep='/', date_format='yyyy-mm-dd'):
        x_list = list()
        y_list = list()
        w_list = list()
        xsqr_list = list()
        ysqr_list = list()
        xy_list = list()
        yr_list = list()
        yr_dist_list = list()
        wx_list = list()
        wxsqr_list = list()
        wy_list = list()
        wxy_list = list()
        wgt_list = list()
        # Loop over the datalist
        for q in datalist2:
            if u == int(q[0]):
                if date_format == 'yyyy-mm-dd':
                    yyyy = int(re.split(date_sep, q[2])[0])
                    mm = int(re.split(date_sep, q[2])[1])
                    dd = int(re.split(date_sep, q[2])[2])
                else:
                    mm = int(re.split(date_sep, q[2])[0])
                    dd = int(re.split(date_sep, q[2])[1])
                    yyyy = int(re.split(date_sep, q[2])[2])
                time = None
                check = leap_year(yyyy)
                if check[0] == "Leap year":
                    time = yyyy + ((((mm-1)*(366.0/12.0)) + dd)/366.0)
                elif check[0] == "Not a leap year":
                    time = yyyy + ((((mm-1)*(365.0/12.0)) + dd)/365.0)
                else:
                    raise CustomException("time is out of bound")
                e = float(q[4])  # Index 4
                x, y, w = time, float(q[3]), 1.0/(e**2)
                xsqr, ysqr, xy = x**2, y**2, x*y
                wx, wxsqr, wy, wxy = w*x, w*x**2, w*y, w*x*y
                # append values
                x_list.append(x)
                y_list.append(y)
                w_list.append(w)
                xsqr_list.append(xsqr)
                ysqr_list.append(ysqr)
                xy_list.append(xy)
                yr_list.append(yyyy)
                yr_dist_list.append((yyyy, y, x))
                wx_list.append(wx)
                wxsqr_list.append(wxsqr)
                wy_list.append(wy)
                wxy_list.append(wxy)
                wgt_list.append((x, y, w))
        res1 = [x_list, y_list, w_list]
        res2 = [xsqr_list, ysqr_list]
        res3 = [xy_list, yr_list, yr_dist_list]
        res4 = [wx_list, wxsqr_list, wy_list, wxy_list, wgt_list]
        return res1, res2, res3, res4

    def write_shoreline_changerate(self, crList, out_csvfile):
        w = open(out_csvfile, 'w')
        w.writelines(
            'TrID,SCE,NSM,EPR,LRR_M,LRR_B,LRR_R2,WLR_M,WLR_B,WLR_WR2\n')
        for ii in crList:
            w.writelines(",".join(str(jj) for jj in ii)+"\n")
        w.close()
        return

    def calc_shoreline_changerate_values(self, infile, out_csvfile, date_sep=None, date_format=None):
        # Read transect data for the temporal shorelines
        datalist2, unique = self.get_data(infile)
        # Loop over the unique transects
        crList = list()
        for u in unique:
            print(u)
            # get params
            res1, res2, res3, res4 = self.extract_shoreline_charngerate_params(datalist2, u, date_sep=date_sep, date_format=date_format)
            x_list, y_list, w_list = res1
            xsqr_list, ysqr_list = res2
            xy_list, yr_list, yr_dist_list = res3
            wx_list, wxsqr_list, wy_list, wxy_list, wgt_list = res4
            # shoreline change-rate calculation
            SCE = self._sce(y_list)
            NSM = self._nsm(yr_list, yr_dist_list)
            EPR = self._epr(yr_list, yr_dist_list, NSM)
            m1, b1, R2 = self._lrr(yr_list, xy_list, x_list,
                                   y_list, xsqr_list, ysqr_list)
            m2, b2, WR2 = self._wlr(w_list, wx_list, wxsqr_list,
                                    wy_list, wxy_list, wgt_list)
            # append values
            crt = list()
            crt.append(u)
            crt.append(round(SCE, 2))
            crt.append(round(NSM, 2))
            crt.append(round(EPR, 2))
            crt.append(round(m1, 2))
            crt.append(round(b1, 2))
            crt.append(round(R2, 2))
            crt.append(round(m2, 2))
            crt.append(round(b2, 2))
            crt.append(round(WR2, 2))
            crList.append(crt)
        # save
        self.write_shoreline_changerate(crList, out_csvfile)
        return


def generate_shoreline_chargerate_values(in_params_csvfile, out_slcr_csvfile, date_sep=None, date_format=None):
    scrt = ShorelineChangerate()
    scrt.calc_shoreline_changerate_values(in_params_csvfile, out_slcr_csvfile, date_sep=date_sep, date_format=date_format)
    return
