import math
import numpy as np
import struct
import xarray as xr

from osgeo import ogr, gdal
from pyproj import Proj
from scipy.spatial import ConvexHull
from scipy.linalg import solve
from app import DEG_TO_KM
from rsgtools.utils import (
    dist_calc,
    flip_line,
    reading_polyline,
    line_slope,
    read_raster_as_array,
    get_shapefile_epsg_code
)
from itertools import product
from sklearn.metrics import mean_squared_error as MSE


# [1]
def bbox_intersect(lyr1Ext, lyr2Ext):
    """
    Bounding Box intersection test.
    """

    # lyr1Ext - reference layer
    b1_x = lyr1Ext[0]
    b1_y = lyr1Ext[2]
    b1_w = lyr1Ext[1] - lyr1Ext[0]  # Horizontal length
    b1_h = lyr1Ext[3] - lyr1Ext[2]  # Vertical length

    # lyr2Ext - clip/intersect layer
    b2_x = lyr2Ext[0]
    b2_y = lyr2Ext[2]
    b2_w = lyr2Ext[1] - lyr2Ext[0]  # horizontal length
    b2_h = lyr2Ext[3] - lyr2Ext[2]  # vertical length

    # query for select object which is inside of the reference extent
    if (b1_x > b2_x + b2_w - 1) or (b1_y > b2_y + b2_h - 1) or (b2_x > b1_x + b1_w - 1) or (b2_y > b1_y + b1_h - 1):
        res = False
    else:
        res = True

    return res


# [2]
def intersect_point_to_line(point, line_start, line_end):
    """
    Find intersect point.
    """

    line_magnitude = dist_calc(line_end, line_start)
    u = ((point[0]-line_start[0])*(line_end[0]-line_start[0])+(point[1] -
         line_start[1])*(line_end[1]-line_start[1]))/(line_magnitude**2)

    # Closest point does not fall within the line segment, take the shorter distance to an endpoint
    if u < 0.00001 or u > 1:
        ix = dist_calc(point, line_start)
        iy = dist_calc(point, line_end)
        if ix > iy:
            res = line_end
        else:
            res = line_start
    else:
        ix = line_start[0]+u*(line_end[0]-line_start[0])
        iy = line_start[1]+u*(line_end[1]-line_start[1])
        res = (ix, iy)

    return res


# [3, 4]
def mid_point(p1, p2, l1, l2):
    iX = p1[0]+((p2[0]-p1[0])*(l1/l2))
    iY = p1[1]+((p2[1]-p1[1])*(l1/l2))
    pts = [iX, iY]
    return pts


def interval_point(objects, nodes, interval):
    points = []
    slopes = []
    for n, i in zip(objects, nodes):
        tDist = 0  # total distance
        vPoint = None  # previous point
        for pnt in i:
            if not isinstance(vPoint, type(None)):
                thisDist = dist_calc(list(vPoint), list(pnt))
                maxAddDist = interval - tDist
                if (tDist+thisDist) > interval:
                    pCnt = int((tDist+thisDist)/interval)
                    for k in range(pCnt):
                        maxAddDist = interval - tDist
                        nPoint = mid_point(
                            vPoint, pnt, maxAddDist, thisDist)
                        slope = line_slope(list(vPoint), nPoint)
                        points.append(nPoint+[n])
                        slopes.append(slope)
                        vPoint = nPoint
                        thisDist = dist_calc(list(vPoint), list(pnt))
                        tDist = 0
                    tDist += thisDist
                else:
                    tDist += thisDist
            else:
                tDist = 0
            vPoint = pnt
    return points, slopes


def fixed_interval_points(infc, interval, flipline=False, save=False, outfc=None, lineslope=False):
    """Generate fixed interval points along polyline.
    """

    # check scr projection and change interval accordingly
    src_epsg_code = get_shapefile_epsg_code(infc)
    if src_epsg_code == 4326:
        interval = interval/DEG_TO_KM/1000.0

    # get nodes and objectids
    if flipline == True:
        openfile = ogr.Open(infc)
        in_layer = openfile.GetLayer(0)
        objects, nodes = flip_line(in_layer)
    else:
        objects, nodes = reading_polyline(infc)

    # generate points
    points, slopes = interval_point(objects, nodes, interval)

    # save if needed
    res = None
    if save == True:
        openfile = ogr.Open(infc)
        in_layer = openfile.GetLayer(0)
        srs = in_layer.GetSpatialRef()
        driver = ogr.GetDriverByName('ESRI Shapefile')
        shapeData = driver.CreateDataSource(outfc)
        out_layer = shapeData.CreateLayer('point', srs, ogr.wkbPoint)

        # create fileds
        lineid = ogr.FieldDefn("LINEID", ogr.OFTInteger)
        out_layer.CreateField(lineid)
        pointid = ogr.FieldDefn("POINTID", ogr.OFTInteger)
        out_layer.CreateField(pointid)
        lyrDef = out_layer.GetLayerDefn()

        # Create point
        for n, p in enumerate(points):
            pt = ogr.Geometry(ogr.wkbPoint)
            pt.SetPoint(0, p[0], p[1])
            feature = ogr.Feature(lyrDef)
            feature.SetGeometry(pt)
            feature.SetField("POINTID", n+1)
            feature.SetField("LINEID", p[2])
            out_layer.CreateFeature(feature)

        # Flush
        shapeData.Destroy()
    elif lineslope == True:
        res = (points, slopes)
    else:
        res = points
    return res


# [5]
def line_intersection(line1, line2):
    """
    Return a (x, y) tuple or None if there is no intersection.
    """
    Ax1, Ay1, Ax2, Ay2 = sum(line1, [])
    Bx1, By1, Bx2, By2 = sum(line2, [])
    # calc difference
    d = (By2 - By1) * (Ax2 - Ax1) - (Bx2 - Bx1) * (Ay2 - Ay1)
    ix = None
    if abs(d) > 0:
        uA = ((Bx2 - Bx1) * (Ay1 - By1) - (By2 - By1) * (Ax1 - Bx1)) / d
        uB = ((Ax2 - Ax1) * (Ay1 - By1) - (Ay2 - Ay1) * (Ax1 - Bx1)) / d
    else:
        ix = None
    # check outer extent
    if not (0 <= uA <= 1 and 0 <= uB <= 1):
        ix = None
    x = Ax1 + uA * (Ax2 - Ax1)
    y = Ay1 + uA * (Ay2 - Ay1)
    ix = x, y
    # return final intersected point (x, y)
    return ix


# [6]
def ray_tracing_method(point, poly):
    """
    Checking if a point or node is inside a polygon.
    """
    x, y = point
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(n+1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


# [7]
def sort_counter_clockwise(points):
    """
    Sort a list of points in counter clockwise.
    """
    centre_x = sum([i[0] for i in points])/len(points)
    centre_y = sum([i[1] for i in points])/len(points)
    angles = [math.atan2(y - centre_y, x - centre_x) for x, y in points]
    cc_indices = sorted(range(len(points)), key=lambda i: angles[i])
    cc_points = [points[i] for i in cc_indices]
    return cc_points


# [8]
def latlon_to_utm(lon, lat):
    """
    Convert lat/long to UTM unit.
    """
    # Compute UTM zone
    utm_zone = int(divmod(lon, 6)[0])+31
    # Defime DD2UTM converter
    dd_to_utm = Proj(proj='utm', zone=utm_zone, ellps='WGS84')
    # Apply the converter
    utm_x, utm_y = dd_to_utm(lon, lat)
    # Add offset if the point in the southern hemisphere
    if lat < 0:
        utm_y = utm_y+10000000
    dd2utm = [utm_x, utm_y, utm_zone]
    return dd2utm


# [9]
def minimum_bounding_rectangle(points):
    """
    Find the minimum area rectangle.
    """
    pi2 = np.pi/2.0
    # get the convex hull for the point list
    vertices_index = list(ConvexHull(points).vertices)
    hull_points = np.array([points[i] for i in vertices_index])
    # calculate edge angles
    edges = hull_points[1:] - hull_points[:-1]
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    angles = np.unique(np.abs(np.mod(angles, pi2)))
    # find rotation matrices
    rotations = np.vstack([np.cos(angles), np.cos(
        angles-pi2), np.cos(angles+pi2), np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))
    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)
    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)
    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)
    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]
    # final rectangle-nodes for close polygon
    rval = np.zeros((5, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)
    rval[4] = np.dot([x1, y2], r)
    return rval


# [10]
def polygon_area(coords):
    """
    Calculate area of polygon given list of coordinates. Assuming coords are in UTM unit.
    """
    # get x and y in vectors
    x = [point[0] for point in coords]
    y = [point[1] for point in coords]
    # shift coordinates
    x_ = x - np.mean(x)
    y_ = y - np.mean(y)
    # calculate area
    correction = x_[-1] * y_[0] - y_[-1] * x_[0]
    main_area = np.dot(x_[:-1], y_[1:]) - np.dot(y_[:-1], x_[1:])
    final_area = 0.5 * np.abs(main_area + correction)
    return final_area


# [11]
def world2Pixel(geoMatrix, x, y):
    ulX = geoMatrix[0]
    ulY = geoMatrix[3]
    xDist = geoMatrix[1]
    pixel = np.round((x - ulX) / xDist).astype(np.int)
    line = np.round((ulY - y) / xDist).astype(np.int)
    return pixel, line


# [12]
def extract_pixel_value(imgfile, point_list):
    """
    Extract pixel value at given point.
    """
    # get raster array
    rasterBands, geoTransform = read_raster_as_array(imgfile)

    pnt_values = []
    for point in point_list:
        # get x and y from point
        mx, my = point  # Coordinates in map units

        # extract value
        px = int((mx - geoTransform[0])/geoTransform[1])  # X pixel
        py = int((my - geoTransform[3])/geoTransform[5])  # Y pixel

        # Assumes 16 bit int aka 'short'
        structVal = rasterBands.ReadRaster(
            px, py, 1, 1, buf_type=gdal.GDT_UInt16)

        # Use the 'short' format code (2 bytes) not int (4 bytes)
        intVal = struct.unpack('h', structVal)

        # print(intVal[0]) # intVal is a tuple, length=1 as we only asked for 1 pixel value
        pnt_values.append([mx, my, intVal[0]])

    return pnt_values


# [13]
def find_wgs2utm_epsg_code(lon: float, lat: float):
    """Based on lat and lng, return best utm epsg-code"""
    utm_band = str((math.floor((lon + 180) / 6) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0'+utm_band
    if lat >= 0:
        epsg_code = '326' + utm_band
        return epsg_code
    epsg_code = '327' + utm_band
    return epsg_code


# [14]
# spatial interpolation algo
def euclidean_dist(x1,x2,y1,y2):
    return ((x1-x2)**2+(y1-y2)**2)**0.5

def nearest_neighbor(df, yy, xx, z_field=None):
    # yy -> Flatten array of Lats (gridded)
    # xx -> Flatten array of Longs (gridded)
    # Nearest Neighbor (NN) 
    array = np.empty((yy.shape[0], xx.shape[0]))
    for i, lat in enumerate(yy):
        for j, lon in enumerate(xx):
            idx = df.apply(lambda row: euclidean_dist(row.LONGITUDE, lon, row.LATITUDE, lat), axis = 1).argmin()
        array[i,j] = df.loc[idx, z_field]
    ds = xr.Dataset(
        {
            z_field: (['lat', 'lon'], array)
        },
        coords={'lat': yy, 'lon': xx}
    )
    return ds

def IDW(df, yy, xx, betta=2, z_field=None):
    # Inverse Distance Weighting (IDW)
    array = np.empty((yy.shape[0], xx.shape[0]))
    for i, lat in enumerate(yy):
        for j, lon in enumerate(xx):
            weights = df.apply(lambda row: euclidean_dist(row.LONGITUDE, lon, row.LATITUDE, lat)**(-betta), axis = 1)
            z = sum(weights*df.z_field)/weights.sum()
            array[i,j] = z
    ds = xr.Dataset(
        {
            z_field: (['lat', 'lon'], array)
        },
        coords={'lat': yy, 'lon': xx}
    )
    return ds

class OrdinaryKriging:
    def __init__(self, lats, lons, values):
        self.lats = lats
        self.lons= lons
        self.values = values
        self.nugget_values = [0, 1, 2, 3, 4]
        self.sill_values = [1, 2, 3, 4, 5]
        self.range_values = [1, 2, 3, 4, 5]

        # Generate all combinations of parameter values
        self.parameter_combinations = list(product(self.nugget_values, self.sill_values, self.range_values))
        self.optimal_pars = None

    def theoretical_variogram(self, h, nugget, sill, r):
        return nugget + (sill-nugget) * (1-np.exp(-3*h/r))

    def euclidean(self, X, Y):
        all_dists, point_dists = [], []
        for x,y in zip(X, Y):
          k = 0
          for k in range(len(X)):
            h = np.linalg.norm(np.array([x, y]) - np.array([X[k], Y[k]]))
            point_dists.append(h)
          all_dists.append(point_dists)
          point_dists = []
        return all_dists

    def gamma(self):
        distances = self.euclidean(self.lats, self.lons)
        differences = np.abs(self.values.reshape(-1,1) - self.values)
        variogram_values = []
        for h in np.unique(distances):
            values_at_h = differences[(distances == h)]
            variogram_values.append(np.mean(values_at_h**2))
        return variogram_values, np.unique(distances)

    def fit(self):
        experimental_variogram, distances = self.gamma()
        fit_metrics = []
        for nugget, sill, range_ in self.parameter_combinations:
            theoretical_variogram_values = self.theoretical_variogram(distances, nugget, sill, range_)
            fit_metric = MSE(experimental_variogram, theoretical_variogram_values)
            fit_metrics.append((nugget, sill, range_, fit_metric))

        self.optimal_pars = min(fit_metrics, key=lambda x: x[3])[:3]

    def predict(self, point):
        points = np.array([(x,y) for x,y in zip(self.lats, self.lons)])
        distances = np.linalg.norm(points - point, axis=1)
        pars = list(self.optimal_pars)
        pars.insert(0, distances)
        weights = self.theoretical_variogram(*pars)
        weights /= np.sum(weights)
        return np.dot(weights, self.values)

def ordinary_kriging(df, yy, xx, z_field=None):
    kriging = OrdinaryKriging(df.LATITUDE.values, df.LONGITUDE.values, df.z_field.values)
    kriging.fit()
    row, grid = [], []
    for lat in yy:
        for lon in xx:
            row.append(kriging.predict(np.array([lat, lon])))
        grid.append(row)
        row=[]
    ds = xr.Dataset(
        {
            z_field: (['lat', 'lon'], grid)
        },
        coords={'lat': yy, 'lon': xx}
    )
    return ds



# Reference:
# [1] https://gis.stackexchange.com/questions/57964/get-vector-features-inside-a-specific-extent
# [2] https://gis.stackexchange.com/questions/396/nearest-neighbor-between-point-layer-and-line-layer
# [3] http://nodedangles.wordpress.com/2011/05/01/quick-dirty-arcpy-batch-splitting-polylines-to-a-specific-length/
# [4] https://github.com/HeyHarry3636/CrossSections_python/blob/master/crossSections_05262016.py
# [5] https://rosettacode.org/wiki/Find_the_intersection_of_two_lines#Python
# [6] https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
# [7] https://stackoverflow.com/questions/69100978/how-to-sort-a-list-of-points-in-clockwise-anti-clockwise-in-python
# [8] https://linuxtut.com/en/4cdf303493d73e24dc14/
# [9] https://stackoverflow.com/questions/13542855/algorithm-to-find-the-minimum-area-rectangle-for-given-points-in-order-to-comput
# [10] https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
# [11] http://stackoverflow.com/questions/13416764/clipping-raster-image-with-a-polygon-suggestion-to-resolve-an-error-related-to
# [12] http://gis.stackexchange.com/questions/46893/how-do-i-get-the-pixel-value-of-a-gdal-raster-under-an-ogr-point-without-numpy
# [13] https://gis.stackexchange.com/questions/269518/auto-select-suitable-utm-zone-based-on-grid-intersection
# [14] https://github.com/alexxxroz/Medium/blob/main/SpatialInterpolation.ipynb

# https://gis.stackexchange.com/questions/392515/create-a-shapefile-from-geometry-with-ogr
# https://www.gis.usu.edu/~chrisg/python/2009/lectures/ospy_slides2.pdf
# http://osgeo-org.1560.x6.nabble.com/satellite-image-processing-in-Python-td3753422.html
# http://lists.osgeo.org/pipermail/gdal-dev/2012-November/034549.html
# http://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html
# http://gis.stackexchange.com/questions/76919/is-it-possible-to-open-rasters-as-array-in-numpy-without-using-another-library
# http://scipy-lectures.github.io/advanced/image_processing/
# http://stackoverflow.com/questions/4842613/merge-lists-that-share-common-elements
# https://gis.stackexchange.com/questions/250077/overlapping-rasters-as-numpy-arrays
# gdal raster datatype: http://ikcest-drr.osgeo.cn/tutorial/k8023
# https://www.quora.com/How-do-I-implement-Otsus-thresholding-in-Python-without-using-OpenCV-and-MATLAB-1
