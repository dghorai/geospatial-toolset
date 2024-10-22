
# Features

### Raster Functionalities

| Functions or Classes                   | Descriptions                                    |
| :------------------------------------- | :---------------------------------------------- |
| clip_raster_by_extent()                | Clip Raster by Extent                           |
| raster_to_point()                      | Convert Rater Pixel to Point Shapefile          |
| dn_to_radiance()                       | Convert Pixel DN Values to Radiance             |
| extract_lulc_class()                   | Export Individual LULC Class                    |
| get_border_pixel_values()              | Extract Border Pixel Values                     |
| group_raster_pixels()                  | Group Homogeneous Pixel Values                  |
| find_sinknflat_dem()                   | Identify Sink/Flat Pixels in DEM                |
| radiance2degree_celsious_temperature() | Convert Radiance to Degree-Celsious Temperature |
| regular_shift_raster()                 | Shift Raster in Different Direction             |
| mosaic_raster()                        | Mosaic GeoTIFF Tiles                            |

### Vector Functionalities

| Functions or Classes           | Descriptions                                       |
| :----------------------------- | :------------------------------------------------- |
| get_cumulative_drainage_area() | Calculate Cumulative Drainage Area                 |
| generate_river_xscl()          | Create Cross-Section Line of River                 |
| generate_grid_boundary()       | Generate Grid Boundary from Point                  |
| get_nearest_point()            | Find Closest Point                                 |
| LineDirectionError()           | Check River Network's Line Direction               |
| GenerateHydroID()              | Generate HydroID of River Network                  |
| CreateGroupID()                | Generate GroupID of River Network                  |
| CreateObjectID()               | Generate ObjectID of River Network                 |
| FnTnodeID()                    | Generate From-Node and To-Node ID of River Network |
| wkt_to_gdf()                   | Convert WKT to Geo-DataFrame                       |
| extract_overlap_polygon()      | Extract Overlap Polygon Geometry                   |
| merge_shapefiles()             | Merge Vector Files                                 |

### Tools for Application

| Functions or Classes               | Descriptions                                                                                   |
| :--------------------------------- | :--------------------------------------------------------------------------------------------- |
| generate_shoreline_raster()        | Shoreline Extraction                                                                           |
| generate_morphometric_parameters() | Morphometric Analysis for Prioritizing Sub-watershed and Management Using Geospatial Technique |
| shoreline_change_analysis() | Digital Shoreline Change-Rate Analysis (example: `notebooks/shoreline_change_rate.ipynb`) |