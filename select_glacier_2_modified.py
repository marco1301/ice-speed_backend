import ogr
import os
import numpy as np
import wikipedia 


#scegli lo shapefile

inShp="/home/studente/tesi_andrea_flavia/17_rgi50_SouthernAndes/17_rgi50_SouthernAndes.shp"
inDriver=ogr.GetDriverByName("ESRI Shapefile")
inDataSource=inDriver.Open(inShp,0)
inLayer=inDataSource.GetLayer(0)
#filtraggio geografico
wkt = "POLYGON ((-74.1 -46.2,-74.1 -47.9, -73.0 -47.9, -73.0 -46.2,-74.1 -46.2))"
inLayer.SetSpatialFilter(ogr.CreateGeometryFromWkt(wkt))
#filtraggio area e tipologia ghiacciai
featureCount=inLayer.GetFeatureCount()
print featureCount
inLayer.SetAttributeFilter("GlacType!='1099'")
featureCount=inLayer.GetFeatureCount()
print featureCount
inLayer.SetAttributeFilter("Area>=20")
featureCount=inLayer.GetFeatureCount()
print featureCount
#creo il layer filtrato
outDriver=ogr.GetDriverByName('ESRI Shapefile')
outDataSource=outDriver.CreateDataSource("/home/studente/tesi_andrea_flavia/shapefiles/Patagonia_Nord5.shp")
outLayer=outDataSource.CopyLayer(inLayer,'Patagonia_Nord5')
path= "/home/studente/tesi_andrea_flavia/shapefiles/"
f=open("/home/studente/tesi_andrea_flavia/databasedef2.txt","w+")
c="Glacier_Name"+"\t"+ "UL_Corner"+"\t"+  "UR_Corner"+"\t"+  "LL_Corner"+"\t"+   "LR_Corner"+"\t"+   "Max_Velocity(m/day)"+"\t"+ "Min_Delta_Time"+"\t"+  "Max_Delta_Time"+"\t"+  "Template_Dim (pixel)"+"\t"+ "Area_Sampling (pixel)"+"\t"+ "Shape_File_Path"
f.write("%s\n" % (c))
#spacchettamento

#####################################################################
for i in range(0, outLayer.GetFeatureCount()):
    feature=outLayer.GetFeature(i)
    a=str(feature.GetField('GLIMSId'))
    stringa_nome= "'"+a+"'"
    print a
    copyDriver=ogr.GetDriverByName('ESRI Shapefile')
    copyDataSource=copyDriver.CreateDataSource("/home/studente/tesi_andrea_flavia/shapefiles/copie/"+ a+".shp")
    copyLayer=copyDataSource.CopyLayer(outLayer,stringa_nome)
    count=copyLayer.GetFeatureCount()
    #print "pre-filtro"
    #print count
    stringa_filtro = "GLIMSId ='"+a+"'"
    copyLayer.SetAttributeFilter(stringa_filtro)
    count=copyLayer.GetFeatureCount()
    #print "dopo filtro"
    #print count
    out2Driver=ogr.GetDriverByName('ESRI Shapefile')
    out2DataSource=out2Driver.CreateDataSource(path + a +".shp")
    #print out2DataSourceclear
    outLayer_single_glac=out2DataSource.CopyLayer(copyLayer,stringa_nome)
    os.remove ("/home/studente/tesi_andrea_flavia/shapefiles/copie/"+a+".dbf" )
    os.remove ("/home/studente/tesi_andrea_flavia/shapefiles/copie/"+a+".prj" )
    os.remove ("/home/studente/tesi_andrea_flavia/shapefiles/copie/"+a+".shx" )
    os.remove ("/home/studente/tesi_andrea_flavia/shapefiles/copie/"+a+".shp" )

    #copyDriver.DeleteDataSource("/home/studente/tesi_andrea_flavia/shapefiles/copie/"+a+".shp")
    extent = outLayer_single_glac.GetExtent()

    NW= extent[3],extent[0]
    NE= extent[3],extent[1]
    SW= extent[2],extent[0]
    SE= extent[2],extent[1]
    name=os.path.basename("/home/studente/tesi_andrea_flavia/shapefiles/"+a+".shp")
    b='{}\t{}{:7.4f}{}{:7.4f}{}\t{}{:7.4f}{}{:7.4f}{}\t{}{:7.4f}{}{:7.4f}{}\t{}{:7.4f}{}{:7.4f}{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(str(a), "[",NW[0], ",", NW[1],"]", "[",NE[0], ",", NE[1],"]","[",SW[0], ",", SW[1],"]", "[",SE[0], ",", SE[1],"]","500","11","48","95","20",str(name))
    with open("/home/studente/tesi_andrea_flavia/databasedef2.txt","w+"):
    	f.write(b)
    	# f.close()



# ghiacciaio = wikipedia.page("San Quintin Glacier")
# coor = ghiacciaio.coordinates
# #print 'Data from ', ghiacciaio.url
# coor = ghiacciaio.coordinates
# lat = round(ghiacciaio.coordinates[0],6)
# lon = round(ghiacciaio.coordinates[1],6)
# print 'lat', lat
# print 'lon', lon


# pointCoord = lon, lat
# fieldName = 'test'
# fieldType = ogr.OFTString
# fieldValue = 'San Quintin'
# outSHPfn = '/home/studente/tesi_andrea_flavia/shapefiles/SanQuintin.shp'

# # Create the output shapefile
# shpDriver = ogr.GetDriverByName("ESRI Shapefile")
# if os.path.exists(outSHPfn):
# 	shpDriver.DeleteDataSource(outSHPfn)
# outDataSource = shpDriver.CreateDataSource(outSHPfn)



# dest_srs=ogr.osr.SpatialReference()
# dest_srs = inLayer.GetSpatialRef()

# outLayer = outDataSource.CreateLayer(outSHPfn, dest_srs, geom_type=ogr.wkbPoint )

# #create point geometry
# point = ogr.Geometry(ogr.wkbPoint)
# point.AddPoint(pointCoord[0],pointCoord[1])

# # create a field
# idField = ogr.FieldDefn(fieldName, fieldType)
# outLayer.CreateField(idField)

# # Create the feature and set values
# featureDefn = outLayer.GetLayerDefn()
# outFeature = ogr.Feature(featureDefn)
# outFeature.SetGeometry(point)
# outFeature.SetField(fieldName, fieldValue)
# outLayer.CreateFeature(outFeature)
# outFeature = None
