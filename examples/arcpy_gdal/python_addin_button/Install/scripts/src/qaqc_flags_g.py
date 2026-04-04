# Import arcpy module
import arcpy
import filter

def XSCL_QAQC_Layer_G(workspace, in_xscl):
    try:
        out_multix_2 = workspace+"\\"+"t_xs_pnt"
        xs_lyr = arcpy.MakeFeatureLayer_management(in_xscl, "xs_Layer", "", "", "")
        infccopy = workspace+"\\"+"t_xs_123"
        point_overlap = workspace+"\\"+"p_overlap"
        infccopy_dup = workspace+"\\"+"t_xs_123_dup"
        model_output2 = workspace+"\\"+"model_output_2x"
        model_output_15 = workspace+"\\"+"t_model_output15" ##

        if arcpy.Exists(out_multix_2):
            arcpy.Delete_management(out_multix_2, "")
        if arcpy.Exists(infccopy):
            arcpy.Delete_management(infccopy, "")
        if arcpy.Exists(point_overlap):
            arcpy.Delete_management(point_overlap, "")
        if arcpy.Exists(infccopy_dup):
            arcpy.Delete_management(infccopy_dup, "")
        if arcpy.Exists(model_output2):
            arcpy.Delete_management(model_output2, "")
        if arcpy.Exists(model_output_15):
            arcpy.Delete_management(model_output_15, "")
        
        arcpy.AddMessage("Multi-intersection Among XSCL")
        arcpy.Intersect_analysis(str(in_xscl)+" #", out_multix_2, "ALL", "0.001 Meters", "POINT")
        arcpy.SelectLayerByLocation_management(xs_lyr, "INTERSECT", out_multix_2, "0.001 Meters", "NEW_SELECTION")
        arcpy.CopyFeatures_management(xs_lyr, infccopy, "", "0", "0", "0")
        arcpy.SelectLayerByAttribute_management(xs_lyr, "CLEAR_SELECTION")
        filter.Filter_Polyline_FC(infccopy, model_output2)

        # Duplicate/exact overlap XS
        ds = arcpy.Describe(in_xscl)
        sr = ds.spatialReference
        
        mid_disc = {}
        cursor = arcpy.SearchCursor(in_xscl)
        for i in cursor:
            mp = i.shape.positionAlongLine(0.50, True).firstPoint
            pnt = (mp.X, mp.Y)
            if not pnt in mid_disc:
                mid_disc[pnt] = []
            mid_disc[pnt].append(-1)

        overlap_list = []
        for k in mid_disc.keys():
            if len(mid_disc[k]) > 1:
                overlap_point = arcpy.Point(k[0], k[1])
                pntGeom = arcpy.PointGeometry(overlap_point)
                overlap_list.append(pntGeom)

        arcpy.CopyFeatures_management(overlap_list, point_overlap)
        arcpy.DefineProjection_management(point_overlap, sr)
        arcpy.SelectLayerByLocation_management(xs_lyr, "INTERSECT", point_overlap, "0.001 Meters", "NEW_SELECTION")
        arcpy.CopyFeatures_management(xs_lyr, infccopy_dup, "", "0", "0", "0")
        arcpy.SelectLayerByAttribute_management(xs_lyr, "CLEAR_SELECTION")

        # Merge
        out1 = int(arcpy.GetCount_management(model_output2)[0])
        out2 = int(arcpy.GetCount_management(infccopy_dup)[0])

        if out1 > 0 and out2 > 0:
            arcpy.Merge_management(model_output2+";"+infccopy_dup+"", model_output_15, "")
        elif out1 > 0 and out2 == 0:
            arcpy.CopyFeatures_management(model_output2, model_output_15, "", "0", "0", "0")
        elif out1 == 0 and out2 > 0:
            arcpy.CopyFeatures_management(infccopy_dup, model_output_15, "", "0", "0", "0")
        else:
            pass
        
        if arcpy.Exists(out_multix_2):
            arcpy.Delete_management(out_multix_2, "")
        if arcpy.Exists(infccopy):
            arcpy.Delete_management(infccopy, "")
        if arcpy.Exists(point_overlap):
            arcpy.Delete_management(point_overlap, "")
        if arcpy.Exists(infccopy_dup):
            arcpy.Delete_management(infccopy_dup, "")
        if arcpy.Exists(model_output2):
            arcpy.Delete_management(model_output2, "")
        if arcpy.Exists(xs_lyr):
            arcpy.Delete_management(xs_lyr, "")

    except Exception as e:
        arcpy.AddMessage(e.message)

    return

if __name__ == '__main__':
    import sys
    XSCL_QAQC_Layer_G(sys.argv[1], sys.argv[2])
