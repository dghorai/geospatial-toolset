# Debabrata Ghorai
# Polygon to centerline conversion

# Import modules
import arcgisscripting
gp = arcgisscripting.create(9.3)
gp.overwriteOutput = True

# Input arguments
infc = gp.GetParameterAsText(0) # r"C:\Work\River_Polygon.gdb\S1_Use"
outLocation = gp.GetParameterAsText(1) # r"C:\Work\poly.gdb"

# Local variables
Densify_Distance = "100 Meters"
gp.workspace = outLocation
tempFeatureClass = infc.split("\\")[-1]

# Processing: Polygon to center line conversion
gp.AddMessage("Feature class reading...")

field = "1"
gp.Densify_edit(infc, "DISTANCE", Densify_Distance, "", "") # Nodes increasing
poinfc = outLocation+"\\poinfc_"+field
gp.AddMessage("Feature to vertices conversion...")
gp.FeatureVerticesToPoints_management(infc, poinfc, "ALL") # Polygon to points conversion
thipoly = outLocation+"\\thipoly_"+field
gp.AddMessage("Thiessen Polygon...")
print("Thiessen polygon")
gp.CreateThiessenPolygons_analysis(poinfc, thipoly, "ALL") # Points to thiessen polygon conversion
thiessen = outLocation+"\\thiessen_"+field
print("Clip")
gp.Clip_analysis(thipoly, thiessen, infc, "") # Clip thiessen polygon wrt to input polygon
thiline = outLocation+"\\thiline_"+field
print("Polygon to line")
gp.FeatureToLine_management(thiessen, thiline, "", "ATTRIBUTES") # Thiessen polygon to line conversion
polyline = outLocation+"\\polyline_"+field
gp.FeatureToLine_management(infc, polyline, "", "ATTRIBUTES") # Input polygon to line conversion
outfc = outLocation+"\\outfc_"+field
print("join")
gp.SpatialJoin_analysis(thiline, polyline, outfc, "JOIN_ONE_TO_ONE", "KEEP_ALL", "", "INTERSECT", "", "") # Spatial joint
OutputFinal = outLocation+"\\outfc1_"+field
gp.Select_analysis(outfc, OutputFinal, "Join_Count = 0") # Select lines if "Joint_Count" is zero
OutDiss = outLocation+"\\outdiss_"+field
gp.Dissolve_management(OutputFinal, OutDiss, "", "", "MULTI_PART", "DISSOLVE_LINES") # Dissolve final selected lines
centerline = outLocation+"\\centerline_"+field
gp.MultipartToSinglepart_management(OutDiss, centerline) # Convert multipart to single part
gp.TrimLine_edit(centerline, "", "DELETE_SHORT") # Remove short lines
gp.AddMessage("Detele temporary files...")
gp.delete(OutDiss) # Delete all temporary files
gp.delete(OutputFinal)
gp.delete(outfc)
gp.delete(polyline)
gp.delete(thiline)
gp.delete(thiessen)
gp.delete(thipoly)
gp.delete(poinfc)
gp.AddMessage("Done!")
