import os
import glob
import geopandas as gpd
import pandas as pd

from shapely.geometry import LineString
from src.utils.geo_utils import extend_line_shapefile, get_shapefile_epsg_code, unlink_files


def get_intersects(baseline, transects, save=False, intersects_points=None):
    line1 = gpd.read_file(baseline)
    line2 = gpd.read_file(transects)
    epsg_code = get_shapefile_epsg_code(transects)
        
    columns_data = []
    geoms = []
    for i in line1.itertuples():
        date = i.DATE
        
        try:
            name = i.NAME
        except:
            name = ''
        
        try:
            e_val = i.E
        except:
            e_val = 0.01
            
        r = i.geometry
        for j in line2.itertuples():
            seqid = j.SeqID
            xg = j.geometry
            l_coords = list(xg.coords)
            # extent at start
            c = LineString(l_coords)
            intersect = r.intersection(c)
            if intersect.geom_type == 'MultiPoint':
                points = [p for p in intersect.geoms]
                for pnt in points:
                    columns_data.append((seqid, date, e_val, name))
                    geoms.append(pnt)
            else:
                columns_data.append((seqid, date, e_val, name))
                geoms.append(intersect)
                
    gdf = gpd.GeoDataFrame(
        columns_data, 
        crs='epsg:'+str(epsg_code),
        geometry=geoms, 
        columns=['SeqID', 'DATE', 'E', 'NAME']
    )
    
    if save:
        gdf.to_file(intersects_points)
    else:
        return gdf
    return


def create_changerate_analysis_points(temp_dir, transects_with_seqid, baseline, shorelines_path, merge_intersect_points):
    # define temporary files
    transects_extended = os.path.join(temp_dir, 'tests', 'results', 'transects_seqid_extd.shp')
    base_and_tr_intx = os.path.join(temp_dir, 'tests', 'results', 'base_with_transects_intx.shp')
    shorelines_and_tr_intx = os.path.join(temp_dir, 'tests', 'results', 'shorelines_with_transects_intx.shp')
    
    # extent transects lines
    extend_line_shapefile(transects_with_seqid, transects_extended, max_snap_dist=25.0, keep_field='SeqID', flip=True)
    # get intersection points between baseline and transaction line
    get_intersects(baseline, transects_extended, save=True, intersects_points=base_and_tr_intx)
    # get intersection points between shorelines and transaction line
    shorelines = glob.glob(os.path.join(shorelines_path, '*.shp'))
    
    gdfs = []
    for shoreline in shorelines:
        gdf = get_intersects(shoreline, transects_extended)
        gdfs.append(gdf)
    
    fgdf = pd.concat(gdfs).pipe(gpd.GeoDataFrame)
    fgdf.reset_index(drop=True, inplace=True)
    fgdf.to_file(shorelines_and_tr_intx)
    
    # merge both the point files
    points1 = gpd.read_file(base_and_tr_intx)
    points2 = gpd.read_file(shorelines_and_tr_intx)
    intersect_points = pd.concat([points1, points2]).pipe(gpd.GeoDataFrame)
    intersect_points.to_file(merge_intersect_points)
    
    # remove temporary files
    unlink_files(is_file=True, file_path=transects_extended)
    unlink_files(is_file=True, file_path=base_and_tr_intx)
    unlink_files(is_file=True, file_path=shorelines_and_tr_intx)
        
    return
