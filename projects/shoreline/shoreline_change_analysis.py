#!/usr/bin/env python
"""
****************************************************************************
* File Name:      shoreline_change_analysis.py
* Created:        April 12, 2024
* Author:         Debabrata Ghorai, Ph.D.
* Purpose:        Digital Shoreline Change Rate Analysis System.
* Description:    Create shoreline transects, generate sequence id of the transects, create change rate analysis inputs, and 
                  calculate shoreline change rate value with different models (SCE, NSM, EPR, LRR_M, LRR_B, LRR_R2, WLR_M, WLR_B, WLR_WR2).
* Research Paper: Ghorai, D., & Bhunia, G. S. (2022). Automatic shoreline detection and its forecast: a case study 
                  on Dr. Abdul Kalam Island in the section of Bay of Bengal. Geocarto International, 37(8), 2273-2292.
                  Available at: https://www.tandfonline.com/doi/abs/10.1080/10106049.2020.1815868
* Revesions:      N/A
****************************************************************************

This project only handle shapefile.

"""


import os

from src.utils.geo_utils import unlink_files
from projects.shoreline.shoreline_baseline_intersects import create_changerate_analysis_points
from projects.shoreline.shoreline_dist_from_baseline import measure_historical_shoreline_distance
from projects.shoreline.shoreline_change_rate import generate_shoreline_chargerate_values


class ShorelineChangeAnalysis:
    """
    baseline: 
        DATE: Null
        NAME: Baseline

    past shorelines:
        DATE: YYYY-MM-DD
        E: Shoreline Uncertainty Value
        
    transects:
        SeqID: Value of SequenceID
    """

    def __init__(self, tmp, offshoreline, baseline,
                shorelines_path, transects, out_csvfile, 
                interval=100, 
                date_sep='-', 
                date_format='yyyy-mm-dd'
        ):
        self.temp_dir = tmp
        self.baseline = baseline
        self.offshoreline = offshoreline
        self.shorelines_path = shorelines_path
        self.x_interval = interval
        self.transects_with_seqid = transects
        self.out_changerate_csvfile = out_csvfile
        self.transects = os.path.join(tmp, transects.replace('.shp', '_temp.shp'))
        self.merge_intersect_points = os.path.join(tmp, 'tests', 'results', 'merge_points.shp')
        self.shoreline_change_params = os.path.join(tmp, 'tests', 'results', 'shoreline_distances.csv')
        self.date_sep = date_sep
        self.date_format = date_format

    def run(self):
        # generate intersects points for shorelines and baseline
        create_changerate_analysis_points(
            self.temp_dir, 
            self.transects_with_seqid, 
            self.baseline, 
            self.shorelines_path, 
            self.merge_intersect_points
        )
        # calculate shoreline distance from baseline
        measure_historical_shoreline_distance(
            self.merge_intersect_points, 
            self.shoreline_change_params
        )
        # calculate shoreline change rate
        generate_shoreline_chargerate_values(
            self.shoreline_change_params, 
            self.out_changerate_csvfile,
            date_sep=self.date_sep, 
            date_format=self.date_format
        )
        # remove temporary files
        unlink_files(is_file=True, file_path=self.transects)
        unlink_files(is_file=True, file_path=self.merge_intersect_points)
        unlink_files(is_file=True, file_path=self.shoreline_change_params)
        return


def generate_shoreline_change_rate(
    tmp=None, 
    offshoreline=None, 
    baseline=None, 
    shorelines_path=None, 
    transects=None, 
    out_csvfile=None, 
    interval=None, 
    date_sep=None, 
    date_format=None
):
    sca = ShorelineChangeAnalysis(
        tmp, 
        offshoreline, 
        baseline, 
        shorelines_path, 
        transects, 
        out_csvfile, 
        interval=interval, 
        date_sep=date_sep, 
        date_format=date_format
    )
    sca.run()
    return
