# Debabrata Ghorai
# Feature class subset by field

# Import modules
import arcgisscripting
gp = arcgisscripting.create()
gp.overwriteOutput = True

# Input arguments
infc = gp.GetParameterAsText(0) # r"C:\Work\Debu\InputData.gdb\test_file"
outLocation = gp.GetParameterAsText(1) # r"C:\Work\Debu\InputData.gdb"
fieldName = gp.GetParameterAsText(2) # "Zone_ID"

# Local variables
gp.workspace = outLocation
delimitedField = gp.AddFieldDelimiters(gp.workspace, fieldName)
tempFeatureClass = infc.split("\\")[-1]

# Processing: Unique feature class count
gp.AddMessage("Feature class counting...")
desc = gp.Describe(infc)
shapefieldname = desc.ShapeFieldName
rows = gp.SearchCursor(infc)

fieldvalues = []
for row in rows:
    field = row.getValue(fieldName)
    fieldvalues.append(field)
uniquefieldval = list(set(fieldvalues))
del row, rows

# Processing: Split feature class by field values
gp.AddMessage("Feature class spliting by field values...")
for ii in uniquefieldval:
    outFeatureClass = tempFeatureClass+"_"+ii
    expression = delimitedField + " = '"+ii+"'"
    gp.AddMessage("Clip feature class name : %s" % outFeatureClass)
    gp.FeatureClassToFeatureClass_conversion(infc, outLocation, outFeatureClass, expression)
del ii, uniquefieldval
gp.AddMessage("Done!")
