# -*- coding: utf-8 -*-
import datetime
import gdal
import numpy as np
import sys, os
import matching_functions as mf
import ogr
import cv2

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pandas
from osgeo import osr
import scipy
#Global variable 

pathresult = "./velocity/data/"

pathresult_filtered = "./velocity/data_filtered/"

shapes_path = "./AREAS/shp/"
Sentinel1_GSD = 10.0 #resolution in meters

#image_results_file = 'IMG-' + img2[:-4] + '.txt'
#terrain_results_file = 'TER-' + img2[:-4] + '.txt'

#import ipdb
#ipdb.set_trace()


class offset_tracking(object):

    ''' Class for handling the off-set tracking procedure over one area between two images'''
    
    def __init__(self, path_img1, path_img2, area_obj, verbose = False):

        #super(offset_tracking, self).__init__()
        #self.img1 = path_img1
        #self.img2 = path_img2

        try:
            self.img1 = gdal.Open(path_img1)
            self.img2 = gdal.Open(path_img2)
        except:
            raise ValueError("ERROR NO VALID IMAGE FORMAT: Please provide a geotif images")

        self.area = area_obj
        self.pathresult = "./velocity/data/"

        self.images_to_be_process = []
        path_img=[path_img1,path_img2]

        for path in path_img:
            path, file = os.path.split(path)
            if file[-4:len(file)] == ".tif": 
                self.images_to_be_process.append(file)
            else:
                print "ERROR NO VALID IMAGE FORMAT: Please provide a geotif images"
                sys.exit()

        self.v = verbose
        if self.v == True: print "\n\t\t\t Off-Set tracking instance has been correctly initialized"

        self.xp_imag=[]
        self.yp_imag=[]
        self.xp_obj=[]
        self.yp_obj=[]
        self.vx = []
        self.vy = []

        self.refine_dx = 0.0
        self.refine_dy = 0.0

    def img_byscl(img, max=255.0):
        imgeq =(255.0*img)/max
        index = np.where(img > 254)
        imgeq[index]= 255
        return np.uint8(imgeq)    


    def refine_coregistration(self):

        img01_band = self.img1.GetRasterBand(1).ReadAsArray(0, 0, self.img1.RasterXSize, self.img1.RasterYSize) 
        img02_band = self.img2.GetRasterBand(1).ReadAsArray(0, 0, self.img2.RasterXSize, self.img2.RasterYSize) 
    
        # Convert images to grayscale
        im1_gray = img01_band.astype('uint8')
        im2_gray = img02_band.astype('uint8')
         
        # Find size of image1
        sz = im1_gray.shape
         
        # Define the motion model
        warp_mode = cv2.MOTION_TRANSLATION
         
        # Define 2x3 or 3x3 matrices and initialize the matrix to identity
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else :
            warp_matrix = np.eye(2, 3, dtype=np.float32)
         
        # Specify the number of iterations.
        number_of_iterations = 5000;
         
        # Specify the threshold of the increment
        # in the correlation coefficient between two iterations
        termination_eps = 1e-10;
         
        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
         
        # Run the ECC algorithm. The results are stored in warp_matrix.
        (cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)
         
        self.refine_dx = warp_matrix[0,2]
        self.refine_dy = warp_matrix[1,2]

        if (abs(self.refine_dx) <1.0 and abs(self.refine_dy) <1.0):
            if self.v == True: print " \n\t\t\t Subpixel coregistration shift: ", self.refine_dx ,  self.refine_dy

        else:
            if self.v == True: print "Subpixel coregistration execeed one pixel: ", self.refine_dx ,  self.refine_dy
            self.refine_dx = 0
            self.refine_dy = 0

    def sampling(self):
    
        # print "Sampling..."
        
        geotransform1 = self.img1.GetGeoTransform()

        # Define pixel_size which equals distance betweens points
        pixel_size = geotransform1[1]*self.area.get_area_sampling()
        
        if self.v == True: print (" \n\t\t\t Selected a regular grid with a "+ str(int(pixel_size)) + "×" +str(int(pixel_size)) +" meters")
        
        source_ds = ogr.Open(shapes_path + self.area.get_shape_path())
        source_layer = source_ds.GetLayer()
        
        x_min = geotransform1[0]
        x_max = geotransform1[0]+geotransform1[1]*self.img1.RasterXSize
        
        y_min = geotransform1[3]+geotransform1[5]*self.img1.RasterYSize
        y_max = geotransform1[3]

        # Create the destination data source
        x_res = int((x_max - x_min) / pixel_size)
        y_res = int((y_max - y_min) / pixel_size)

        target_ds = gdal.GetDriverByName('GTiff').Create('temp.tif', x_res, y_res, gdal.GDT_Byte)
        target_ds.SetProjection(self.img1.GetProjection())
        target_ds.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
        band = target_ds.GetRasterBand(1)
        band.SetNoDataValue(255)

        # Rasterize
        gdal.RasterizeLayer(target_ds, [1], source_layer, burn_values=[1])

        # Read as array
        array = band.ReadAsArray()

        # Convert array to point coordinates
        count = 0
        roadList = np.where(array == 1)
        multipoint = ogr.Geometry(ogr.wkbMultiPoint)
        
     
        for k in xrange(len(roadList[0])):
            indexY = roadList[0][count]
            indexX = roadList[1][count]
        
            Xcoord = indexX*self.area.get_area_sampling()
            Xobject=geotransform1[0]+indexX*pixel_size

            Ycoord = indexY*self.area.get_area_sampling()
            Yobject=geotransform1[3]+indexY*(-pixel_size)
        
            nl="\n"
            
            self.xp_obj.append(Xobject)
            self.yp_obj.append(Yobject)
            
            self.xp_imag.append(Xcoord)
            self.yp_imag.append(Ycoord) 

            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint(Xcoord, Ycoord)
            multipoint.AddGeometry(point)
            count += 1
            
        if self.v == True: print "\t\t\t Number of points: ", len(self.xp_imag)


        with open(pathresult+"/"+"terrain_coord.txt","w") as grid:
            nl="\n"
            for i in xrange(len(self.xp_obj)):
                grid.writelines(  (str(self.xp_obj[i]) +","+"\t"+ str(self.yp_obj[i]),nl  )) 
        
        with open(pathresult+"/"+"image_coord.txt","w") as grid:
            nl="\n"
            for i in xrange(len(self.xp_imag)):
                grid.writelines((str(self.xp_imag[i]) +","+"\t"+ str(self.yp_imag[i]),nl))      
                                                                                                                      

    def matching_area(self):
      
        ######### Compute Displacement ##########
        Temp_DIMI = self.area.get_template_dim() 
        Temp_DIMJ = self.area.get_template_dim() 

        board = int(((int(self.area.get_max_velocity())/30.0)/Sentinel1_GSD)*self.deltatime.days)
        
        if self.v == True: print "\t\t\t Expected maximum glacier velocity selected: ", str(self.area.get_max_velocity())+ " meters/month"
        
        if self.v == True: print "\t\t\t Template: ", str(Temp_DIMI)+"x"+ str(Temp_DIMJ)+ " pixel ", "Search edge: ",  str(board)+" pixel"
        
        print self.area.get_param_string()

        board_1 = 1
        DS=[]
            
        for k in range(len(self.xp_imag)):
            x=self.xp_imag[k]
            y=self.yp_imag[k]
                
                
            img01_band = self.img1.GetRasterBand(1).ReadAsArray(int(x)-Temp_DIMI/2, int(y)-Temp_DIMJ/2, Temp_DIMI, Temp_DIMJ) #template
            img02_band = self.img2.GetRasterBand(1).ReadAsArray(int(x)-Temp_DIMI/2-board, int(y)-Temp_DIMJ/2-board, Temp_DIMI+2*board, Temp_DIMJ+2*board)   #search area 0
                        
            i,j, maxcc =mf.template_match(img01_band.astype('uint8'), img02_band.astype('uint8'),mlx=2,mly=2, show = False)
            Dx = int(i-Temp_DIMI/2.0-board)
            Dy = int(j-Temp_DIMJ/2.0-board)
                
            # Subpixel refinement 
            img02_band = self.img2.GetRasterBand(1).ReadAsArray(int(x+Dx)-Temp_DIMI/2-board_1, int(y+Dy)-Temp_DIMJ/2-board_1, Temp_DIMI+2*board_1, Temp_DIMJ+2*board_1)   #search area 1
   
            i,j, maxcc =mf.template_match(img01_band.astype('uint8'), img02_band.astype('uint8'),mlx=4,mly=4, show = False)
                                   
            Dx1= i-Temp_DIMI/2.0-board_1
            Dy1= j-Temp_DIMJ/2.0-board_1

            # CONTROLLARE REFINE SUBPIXEL
            DX=Dx+Dx1-self.refine_dx
            DY=Dy+Dy1-self.refine_dy

            D=np.sqrt((DX**2)*self.sf*self.sf+(DY**2)*self.sf*self.sf)

            self.vx.append(DX)
            self.vy.append(DY)

            DS.append([int(x),int(y),DX,DY,maxcc,Dx,Dy,Dx1,Dy1,np.sqrt(DX**2+DY**2),(maxcc), (DX*self.sf), (DY*self.sf), D]) #CONTROLLARE CAMPI DI STAMPA MAXCC 
  
        
        image_results_file = 'IMG-' + self.images_to_be_process[1][:-4] + '.txt'
        terrain_results_file = 'TER-' + self.images_to_be_process[1][:-4] + '.txt'
        
        with open(pathresult+"/"+image_results_file,"w") as grid_img:

            grid_img.write("x, "+"y, "+"dx, "+"dy,"+"MaxCC, "+"dx0,"+"dy0, "+"dx1, "+"dy1, "+"||S||, "+"MaxCC^-1,"+"dx(m),"+"dy(m),"+"||s||(m)\n")
            count1=0
                
            for item in DS:
                grid_img.writelines(str(item[0])+"\t"+str(item[1])+"\t"+str(item[2])+"\t"+str(item[3])+"\t"+str(item[4])+"\t"+str(item[5])+"\t"+str(item[6])+"\t"+str(item[7])+"\t"+str(item[8])+"\t"+str(item[9])+"\t"+str(item[10])+"\t"+str(item[11])+"\t"+str(item[12])+"\t"+str(item[13])+"\n") 
                    
                count1 +=1
            

        #Conversion from utm to geo --> INSERIRE CONTROLLO SR CON SHAPE
        from pyproj import Proj, transform     

        Lon = self.area.get_corners()[0][0]
        Lat = self.area.get_corners()[0][1]

        UTM_str = ""

        if Lat > 0: 
            UTM_str = "epsg:326"+str(np.int(np.ceil((Lon+180)/6)))
        else:
            UTM_str = "epsg:327"+str(np.int(np.ceil((Lon+180)/6)))

        inpr = Proj(init=UTM_str) 
        outpr = Proj(init='epsg:4326')  

        Geo_Coord =  transform(inpr,outpr,self.xp_obj,self.yp_obj)  

        Angle = np.arctan2(self.vy,self.vx)       
        Angle = 180*Angle/np.pi  
            
        with open(pathresult+"/"+terrain_results_file,"w") as grid_ter:
            grid_ter.write('{:>12}\t{:>12}\t{:>8}\t{:>10}\t{:>11}\t{:>8}\t{:>8}\t{:>8}\n'.format(
                "Lon(°)", "Lat(°)", "Ang(°)", "E(m)", "N(m)", "dx(m)","dy(m)","||s||(m)"))
            count2=0
            
            for item in DS:
                rstr = '{:12.7f}\t{:12.7f}\t{:8.3f}\t{:10.2f}\t{:11.2f}\t{:8.3f}\t{:8.3f}\t{:8.3f}\n'.format(    
                    Geo_Coord[0][count2], Geo_Coord[1][count2], Angle[count2], self.xp_obj[count2], self.yp_obj[count2],item[11], item[12], item[13])
               
                grid_ter.writelines(rstr)

                count2 +=1 
     

        if self.v == True: 
            print "\n\t\t\t Output files: " 
            print "\t\t\t ",  image_results_file  
            print "\t\t\t ",  terrain_results_file
    

    def spatial_filter(self):


        terrain_results_file = 'TER-' + self.images_to_be_process[1][:-4] + '.txt' 
        image_results_file = 'IMG-' + self.images_to_be_process[1][:-4] + '.txt'
        #pathresult global variable
        
        data=np.genfromtxt(str(pathresult+"/"+image_results_file), skip_header = 1, dtype= "float32" ) 
        data_img=np.genfromtxt(str(pathresult+"/"+terrain_results_file), skip_header = 1)
        img_cord=data_img[:,(0,1)]
        ter_cord=data[:,(3,4)]
        azim=data[:,2]*((np.pi/180))

        img=self.img1
        
        #azimuth matrix
        Est_list=[]
        Nord_list=[]

        for i in xrange(0,img.RasterYSize):
            Est_list.append(i)
        for j in xrange(0,img.RasterXSize):
            Nord_list.append(j)
        df=pandas.DataFrame(np.zeros((len(Nord_list),len(Est_list))),index=Nord_list,columns=Est_list)
        i=0

        for row in img_cord:
            est=row[0]
            nord=row[1]
            df.loc[est,nord]=azim[i]
            i+=1

        geotransform1 = img.GetGeoTransform()
        ymin=geotransform1[3]+geotransform1[5]*max(df.columns)
        ymax=geotransform1[3]
        xmin = geotransform1[0]
        xmax = geotransform1[0]+geotransform1[1]*max(df.index)

        Matrice=df.as_matrix()
        Matrice=np.array(Matrice)
        nrows,ncols = np.shape(Matrice)
        geotransform=(xmin,ncols,0,ymax,0, -nrows)

        ds = gdal.GetDriverByName('MEM').Create('', ncols, nrows, 1, gdal.GDT_Float32)
        ds.GetRasterBand(1).WriteArray(Matrice)


       #filter dimension
        board = 35
        Temp_tot = 71 #pixel
       
        xp_img =img_cord[:,0]
        yp_img =img_cord[:,1]
        list_dev=[]
        listaxy=[]
        list_elem=[]
        for k in xrange(len(xp_img)):
            x=xp_img[k]
            y=yp_img[k]
            azim_img_band=ds.GetRasterBand(1).ReadAsArray(int(y)-board, int(x)-board, Temp_tot, Temp_tot)
            azim_index = np.where(azim_img_band != 0)
            azim_values = azim_img_band[azim_index[0], azim_index[1]]
            neg_ind = np.where(azim_values<0)
            azim_values[neg_ind] = azim_values[neg_ind]+2*np.pi 
            asc=len(azim_values)
            list_elem.append(asc)
            c=np.mean(np.cos(azim_values))
            s=np.mean(np.sin(azim_values))
            r=np.sqrt(c**2+s**2) #
            #dev_st=cv2.meanStdDev(azim_values)
            #dev_st=scipy.stats.circstd(azim_values)
            list_dev.append(r)

            #list_dev.append(dev_st[1][0][0])
            # deviazione=dev_st[1][0][0]
            if  r<0.7:
                coordx = azim_index[0] + (int(x)- board)
                coordy = azim_index[1] + (int(y)- board)
                j=0
                for i in coordx:
                    indice= np.where(np.logical_and( img_cord[:,0]==i, img_cord[:,1] == coordy[j]))
                    if len(indice[0])>0:
                        listaxy.append(int(indice[0][0]))
                    j +=1
        listaxypro=listaxy
        listaxypro.sort()
        listaxy_final=list(set(listaxypro))
        copy_data=data
        copy_data[listaxy_final,5] = 0#1
        copy_data[listaxy_final,6] = 0#1
        copy_data[listaxy,7] = 0#1
        copy_data_img = img_cord
        copy_data_img[listaxy_final, (0)]=0#1
        copy_data_img[listaxy_final, (1)]=0#1

        terrain_filtered_results_file = 'TER-' + self.images_to_be_process[1][:-4] +'filtered'+'.txt'

        with open(pathresult_filtered+"/"+terrain_filtered_results_file,"w") as grid_ter:
            grid_ter.write('{:>12}\t{:>12}\t{:>8}\t{:>10}\t{:>11}\t{:>8}\t{:>8}\t{:>8}\n'.format(
                "Lon(°)", "Lat(°)", "Ang(°)", "E(m)", "N(m)", "dx(m)","dy(m)","||s||(m)"))
            for item in copy_data:
                rstr = '{:12.7f}\t{:12.7f}\t{:8.3f}\t{:10.2f}\t{:11.2f}\t{:8.3f}\t{:8.3f}\t{:8.3f}\n'.format(    
                    item[0], item[1], item[2], item[3], item[4], item[5], item[6], item[7])
                grid_ter.writelines(rstr)

        if self.v == True: 
            print "\n\t\t\t Output filtered files: " 
             
            print "\t\t\t ",  terrain_filtered_results_file


    def stampa(self, risultati,soglia_sup,soglia_inf,sf, graph_results_file):
            
        soglia_sup=int(soglia_sup)
        soglia_inf=int(soglia_inf)
        
        H=self.img1.RasterYSize
        W=self.img1.RasterXSize

        plt.ylim([0,H])   #intervallo sulle y
        plt.xlim([0, W])  #intervallo sulle x
        
        
        data=np.loadtxt(risultati,skiprows=1)
        a=data    
        modulo_spostamenti=np.sqrt((a[:,2]**2)*sf*sf+(a[:,3]**2)*sf*sf)
        
        #rimuovo dal grafico le deformazioni con modulo inferiore/superiore al valore della soglia
        limite_inf = np.where(modulo_spostamenti>soglia_sup)[0]
        modulo_spostamenti=np.delete(modulo_spostamenti,limite_inf,axis=0)
        
        limite_sup=np.where(modulo_spostamenti<soglia_inf)[0]
        modulo_spostamenti=np.delete(modulo_spostamenti,limite_sup,axis=0)
           
        a = np.delete(a, limite_inf, axis=0) 
        a = np.delete(a,limite_sup,axis=0)
        
        nz = mcolors.Normalize(vmin=soglia_inf,vmax=soglia_sup)
        
        fig = plt.figure(str(data[0])+" - "+str(data[1]))
        ax  = fig.add_subplot(111)
        
        #ax.set_prop_cycle(['red', 'black', 'yellow'])
        
        #ax.set_color_cycle(['red', 'black', 'yellow'])

    	# così imposto la stessa scala su x e y
        plt.gca().set_aspect('equal', adjustable='box')
         
        plt.ylabel('pixel')
        plt.xlabel('pixel')
        
        img_band=self.img1.GetRasterBand(1).ReadAsArray(0,0,W,H) 
        img_band_int=np.int_(img_band)
        plt.imshow(img_band_int, cmap=plt.cm.gray)
     	
     	# così disegno le freccette
        plt.quiver(a[:,0], a[:,1], (a[:,2]*sf)/modulo_spostamenti, (sf*a[:,3])/modulo_spostamenti,angles='xy', scale=50,color=cm.jet(nz(modulo_spostamenti)))
      	
      	# così imposto la colorbar
        divider=make_axes_locatable(ax)   
        cax = divider.append_axes("right", size="5%", pad=0.05) 
        cb = mcolorbar.ColorbarBase(cax, cmap=cm.jet, norm=nz) 
        cb.set_clim(soglia_inf, soglia_sup)# 
        cb.set_label('meters/month')
        
        plt.savefig(pathresult+"/"+graph_results_file,dpi=250,bbox_inches='tight')
             
        fig.clear()
        plt.close()             
               
                                    
    def analyse_pair(self):
                           
        #######lista delle date per nominare i risultati####
        date=[]
        for file in self.images_to_be_process:
            a= file[17:25]  
            date.append(str(a))
      
        a_ar=[]
        m_ar=[]
        d_ar=[]

        for item in date:
            anno=int(item[0:4])
            mese=int(item[4:6])
            giorno=int(item[6:8])
            a_ar.append(anno)
            m_ar.append(mese)
            d_ar.append(giorno)
        
                        
        #####lista delle date per calcolare lo spam temporale####
        list_dat=range(len(self.images_to_be_process))
        for i in range(len(self.images_to_be_process)):
            list_dat[i]=datetime.datetime(a_ar[i],m_ar[i],d_ar[i])
        
        list_dat.sort()
                
        self.deltatime = list_dat[1]-list_dat[0]

        if self.v == True: print "\t\t\t Day interval: ", self.deltatime.days     

        self.sf=(10.0*30)/self.deltatime.days     

        self.sampling()

        self.matching_area()
        self.spatial_filter()
    def print_graph(self, soglia_inf = 0, soglia_sup = 600):   
        
        soglia_sup = self.area.get_max_velocity()

        image_results_file = 'IMG-' + self.images_to_be_process[1][:-4] + '.txt'
        risultati = str(pathresult+"/"+image_results_file)

        graph_results_file = 'graph-' + self.images_to_be_process[1][:-4] + '.png'
        self.stampa(risultati, soglia_sup, soglia_inf, self.sf, graph_results_file)
                                          
                                                        
                                                            
                                                                