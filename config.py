# -*- coding: utf-8 -*-
"""
Created on Fri Apr 5 19:26:11 2024

@author: Debabrata Ghorai, Ph.D.

Config Dataclass - Define variables settings.
The main script can read and use these variables.

"""

import os

from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class PointConfig:
    inpoint: Path
    single_point: tuple
    point_offset: float


@dataclass(frozen=True)
class PolylineConfig:
    inline: Path
    fnodeid: str
    tnodeid: str
    lineid: str
    seqid: str
    is_xscl_available: bool
    outxsclline: Path
    uidfield: str
    objectidfield: str
    linefield: str
    interval: int
    offset: int


@dataclass(frozen=True)
class PolygonConfig:
    inpolygon: Path
    cumfield_name: str
    out_grid_polygon: Path


@dataclass(frozen=True)
class VectordataConfig:
    vector_outfile: Path
    vector_file_dir: Path
    vector_geometry_type: str


@dataclass(frozen=True)
class RasterdataConfig:
    raster_infile: Path
    polygon_infile: Path
    clip_raster_file: Path
    out_point_file: Path
    output_folder: Path
    lmax_list: list
    lmin_list: list
    qcal_min: float
    qcal_max: float
