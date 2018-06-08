import sys
import os
import io
import time
from os.path import join
from datetime import datetime
from datetime import timedelta
import dateutil.relativedelta as relativedelta   

from google.cloud import storage
from functions.storage import upload_blob, download_blob

bucket_name = 'icespeed-team'
storage_client = storage.Client()
bucket = storage_client.get_bucket(bucket_name)

time_buffer_size = 90 #days
start_date = datetime.strptime("2015-01-01", '%Y-%m-%d')
end_date = datetime.today()

import plotly as pl
import plotly.graph_objs as go
import numpy as np

import areas_functions as af

import ipdb

import numpy as np
import plotly as pl
import plotly.figure_factory as ff 


import statsmodels.api as sm





import gdal
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable





#define date interval 
t1 = start_date
velocity_interval =[]

while t1 < end_date:
    velocity_interval.append([t1, t1 + relativedelta.relativedelta(months=+2)])
    t1 = t1+ relativedelta.relativedelta(months = +2)

#####################################

#funziona con un solo ghiacciaio





def compute_ti(observed_interval_ori):

    observed_interval = observed_interval_ori.copy()

    dti = []

    for interval in velocity_interval:
        
        if interval[0] <= observed_interval[0] < interval[1]: 

            if observed_interval[1] <= interval[1]:
                dt = observed_interval[1]-observed_interval[0]
            else:
                dt = interval[1]-observed_interval[0]
                observed_interval[0] = interval[1]

            dti.append(dt.days)
        
        else:
            dti.append(0.0)
    
    return dti


def point_least_square(point_data_raw, verbose = False):


    # input listone_risultati[400][2:]

    point_data = point_data_raw.reshape(len(point_data_raw)/3, 3)

    N = point_data.shape[0]
    M = len(velocity_interval)

    A = np.zeros((2*N,2*M))
    B = np.zeros(2*N)

    #Ba = np.zeros(0)
    #ipdb.set_trace()

    for i in xrange(N):

        #print i
        A_row = compute_ti(point_data[i,0])

        delta_t = point_data[i,0][1]-point_data[i,0][0]

        for j in xrange(M):
            A[2*i,j] = A_row[j] 
            A[2*i+1, M+j] = A_row[j]


        B[2*i] = point_data[i,1]* delta_t.days

        B[2*i+1] = point_data[i,2]* delta_t.days
        
#        Bv = np.concatenate((Bv, [point_data[i,1]* delta_t.days]))

#        Ba = np.concatenate((Ba, [point_data[i,2]* delta_t.days]))
    
        #ipdb.set_trace()

    #AA = A.reshape((N,M))

    #ipdb.set_trace()
    #sm.RLM or sm.WLS? essere o non essere robusti?

    if verbose == True: print "VELOCITY RESULTS: \n\n"
    mod_wls = sm.RLM(B, A, weights=1)
    res_wls = mod_wls.fit()
    if verbose == True: print res_wls.summary()


#    if verbose == True: print "AZIMUTH RESULTS: \n\n"
#    mod_wls1 = sm.RLM(Ba, AA, weights=1)
#    res_wls1 = mod_wls1.fit()
#    if verbose == True: print res_wls1.summary()


    #ipdb.set_trace()


    return res_wls.params[0:M], res_wls.params[M:2*M-1]


def stampa(img,soglia_sup,soglia_inf,sf, graph_results_date, x, y, vx, vy):
    soglia_sup= int(soglia_sup)
    soglia_inf = int(soglia_inf)
        
    H=img.RasterYSize
    W=img.RasterXSize

    
        
    #data=np.loadtxt(risultati,skiprows=1)
         
    modulo_spostamenti=np.sqrt((vx**2)*sf*sf+(vy**2)*sf*sf)
        
    #rimuovo dal grafico le deformazioni con modulo inferiore/superiore al valore della soglia
    limite_inf = np.where(modulo_spostamenti>soglia_sup)[0]
    modulo_spostamenti=np.delete(modulo_spostamenti,limite_inf,axis=0)
        
    limite_sup=np.where(modulo_spostamenti<soglia_inf)[0]
    modulo_spostamenti=np.delete(modulo_spostamenti,limite_sup,axis=0)
            
    x = np.delete(x, limite_inf, axis=0) 
    x = np.delete(x,limite_sup,axis=0)

    y = np.delete(y,limite_inf,axis=0)
    y = np.delete(y,limite_sup,axis=0)

        
    vx = np.delete(vx, limite_inf, axis=0)
    vx = np.delete(vx, limite_sup, axis=0)

    vy = np.delete(vy,limite_inf,axis=0)
    vy = np.delete(vy,limite_sup,axis=0)


        
    nz = mcolors.Normalize(vmin=soglia_inf,vmax=soglia_sup)
        
    fig = plt.figure(str(data[0])+" - "+str(data[1]))
    ax  = fig.add_subplot(111)
    #ax.set_prop_cycle(['red', 'black', 'yellow'])
    #ax.set_color_cycle(['red', 'black', 'yellow'])
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.ylabel('Northing (m)')
    plt.xlabel('Easting (m)')
        
    img_band=img.GetRasterBand(1).ReadAsArray(0,0,W,H) 
    img_band_int=np.int_(img_band)

    gt = img.GetGeoTransform()

    extent = (gt[0], gt[0] + img.RasterXSize * gt[1], gt[3] + img.RasterYSize * gt[5], gt[3])
    plt.imshow(img_band_int, extent=extent, origin='upper', cmap=plt.cm.gray)
        
    
    #ipdb.set_trace()
    plt.quiver(x, y, (vx*sf)/modulo_spostamenti, (sf*vy)/modulo_spostamenti,angles='xy', scale=50,color=cm.jet(nz(modulo_spostamenti)))
    plt.ylim([4742500, 4760000])   #intervallo sulle y
    plt.xlim([613000,637000])  #intervallo sulle x    
    
    divider=make_axes_locatable(ax)   
    cax = divider.append_axes("right", size="5%", pad=0.05) 
    cb = mcolorbar.ColorbarBase(cax, cmap=cm.jet, norm=nz) 
    cb.set_clim(soglia_inf, soglia_sup)# 
    cb.set_label('meters/month')
        
    plt.savefig(graph_results_date,dpi=250,bbox_inches='tight')
             
    fig.clear()
    plt.close()    





print('\n--------------------------------')
print('   Image statistics computation ')
print('---------------------------------\n')


areas = af.Areas_from_txt('./AREAS/database.txt')

passings = ['ASCENDING', 'DESCENDING'] #mettere solo ascending 'DESCENDING'

for area in areas[:]: #funziona con su un solo ghiacciaio

    a = 1
    figs = []
    dates = []
    #central_date[]

    for passing in passings:

        blobs = bucket.list_blobs(prefix=join(area.get_glacier_name(), passing, 'RESULTS/S'), delimiter=None)

        for blob in blobs:
            
            names = blob.name.split("/")

            if names[-1][0:3] == "TER" and names[-1][-4:] == ".txt":
                date1 = datetime.strptime(names[-2].split("_")[4], '%Y%m%dT%H%M%S')
                date2 = datetime.strptime(names[-1].split("_")[4], '%Y%m%dT%H%M%S')
                dates.append([date1, date2])

                #################
                if not os.path.exists( "./velocity/data/"+area.get_glacier_name()+"_TER_"+str(date1.date())+"_"+str(date2.date())):
                    download_blob(bucket_name, blob.name,  "./velocity/data/"+area.get_glacier_name()+"_TER_"+str(date1.date())+"_"+str(date2.date())) 
                ################

        dates.sort()

        for date in dates:
        	nums = [a,a]
        	a = a+1
        	figs.append(go.Scatter(x=date, y=nums))

    pl.offline.plot(figs,filename= area.get_glacier_name()+".html")
    figs = 0


    T0 = start_date
    T1 = start_date+ timedelta(days=time_buffer_size)


    #initialize resutl array with x,y coordinates 
    listone_risultati = []
    first_data = np.genfromtxt("./velocity/data/"+area.get_glacier_name()+"_TER_"+str(dates[0][0].date())+"_"+str(dates[0][1].date()), skip_header = 1)

    x = first_data[:,3]  
    y = first_data[:,4]

    for j in xrange(len(x)):
        listone_risultati.append([x[j],y[j]])
    #ipdb.set_trace()
    
    # Analyze stack considering the buffer period 

    while(T1 < end_date+timedelta(days=time_buffer_size/2)):

        print "\t Period from ", T0, " to ", T1

        velocity_data = []

        good_dates = []

        count = 0

        for date in dates:
            
            central_date = date[0] + (date[1]-date[0])/2

            if central_date > T0 and central_date < T1: 
                #print "TER_"+str(date[0].date())+"_"+str(date[1].date())

                velocity_data.append(np.genfromtxt("./velocity/data/"+area.get_glacier_name()+"_TER_"+str(date[0].date())+"_"+str(date[1].date()), skip_header = 1))
                good_dates.append(date)
                count += 1

        print "\t\t Number of observation in the period: ", count

#         if count > 0:
#             velocity_data = np.array(velocity_data)
#             good_dates = np.array(good_dates)
#             azimuth = velocity_data[:,:,2]
#             azimuth[np.where(azimuth<0)] += 360
#             velocity_data[:,:,2] = azimuth
#             sizes = np.shape(velocity_data)
#             vx = []
#             vy = []
#             mod = []
#             for i in xrange(sizes[1]):            #sizes = (5, 1767, 8) (osserv, iesimo punto, n colonna)
#                 vel_point = velocity_data[:,i,5] #dx 
#                 azi_point = velocity_data[:,i,6] #dy vedere i txt dei risultati 
#                 vel_med = np.median(vel_point) #sul punto
#                 azi_med = np.median(azi_point)
#                 vel_std = np.std(vel_point)
#                 azi_std = np.std(azi_point)
#                 vel_NMAD = 1.4826*np.median(np.abs(vel_point-vel_med))
#                 azi_NMAD = 1.4826*np.median(np.abs(azi_point-azi_med))
#                 mod.append(vel_med)
#                 if azi_NMAD < 25 and vel_NMAD < 25:
#                     vx.append(vel_med)
#                     vy.append(-azi_med)
#                 else:
#                     vx.append(0)
#                     vy.append(0)
#                 good_values_azi = np.where( np.abs(velocity_data[:,i,6]-azi_med) < 2*azi_NMAD  )
#                 good_values_vel = np.where( np.abs(velocity_data[:,i,5]-vel_med) < 2*vel_NMAD  )
#                 good_values = np.intersect1d(good_values_azi[0], good_values_vel[0])
#                 for indx in good_values:
#                     listone_risultati[i].append(good_dates[indx])
#                     listone_risultati[i].append(vel_point[indx])
#                     listone_risultati[i].append(azi_point[indx])


#         #global good values min,max, mean, stddev

#         #scatter |vel| vs  

#             #print "{:8.3f}{:8.3f}{:8.3f}{:8.3f}{:8.3f}{:8.3f}".format(vel_med, vel_std,vel_NMAD, azi_med, azi_std, azi_NMAD)  #, velocity_data[:,i,7], velocity_data[:,i,2]

#         #END FOR 

#         fig = ff.create_quiver(x,y,vx,vy, scale= 30.0, marker=dict(size='16', 
#         color = mod, #set color equal to a variable
#         colorscale='Viridis',
#         showscale=True))

#         #pl.offline.plot(fig, filename=area.get_glacier_name()+'_plot_'+str(T0.date())+"_"+str(T1.date())+'.html')    

        T0 = T1
        T1 = T1 + timedelta(time_buffer_size)

#     #END WHILE 

#     print('\n-----------------------------')
#     print('   Least Square Adj Started')
#     print('------------------------------\n ')

#     with open("results_test"+area.get_glacier_name()+str(area.get_max_velocity())+".txt","w") as file:

#         Num = len(listone_risultati)
#         count = 0.0 

#         print "\t Number of grid point: ", Num

#         print "\n\t Number of computed velocity interval: ", len(velocity_interval), "\n"

#         for results_punto in listone_risultati:

#             if len(results_punto) > 2:
#                 try:
#                     Vel, Azi = point_least_square(np.array(results_punto[2:]).copy(), verbose=False)
                
#                     file.write('{:12.3f}\t{:12.3f}\t'.format(results_punto[0], results_punto[1]))

#                     for i in xrange(len(Vel)-1):
#                         file.write('{:8.3f}\t{:8.3f}\t'.format(Vel[i], Azi[i]))

#                     file.write('\n')    

#                 except:
#                     print "Invalid point"

                
#             count += 1.0

#             print "\t Percent complete: {:5.1f}".format(count/Num*100), " \r",

#             #print("\t Percent complete: {:5.1f}".format(count/Num*100), end='\r')
#             sys.stdout.flush()


    



#     ######## GRAFICI
# data = np.genfromtxt("results_test"+area.get_glacier_name()+str(area.get_max_velocity())+".txt", skip_header=0)  

# x = data[:,0]
# y = data[:,1]
    

# list_img=[]
# blobs = bucket.list_blobs(prefix=join(area.get_glacier_name(), passing, 'S') , delimiter=None)
# for blob in blobs:
#     names_img=blob.name.split("/")
#     img=str(names_img[2])
#     list_img.append(img)

    
# img1=list_img[1]

# download_blob(bucket_name, join(area.get_glacier_name(), passing, img1), join('velocity/data', img1))

# img_SAR = gdal.Open( join('velocity/data', img1) )
# geotransform = img_SAR.GetGeoTransform()

# graph_results = "./velocity/data"
#  #conversione in coord immagine
# #x_im = x/geotransform[1] - geotransform[0] / geotransform[1]
# #y_im = y/ geotransform[5] - geotransform[3] / geotransform[5]

# for ii in xrange((len(data[0,:])-2)/2):

#     vx = data[:,2*ii+2]
#     vy = -data[:,2*ii+3]
       
        

#     t1 = "{:%Y_%m_%d}".format(velocity_interval[ii][0])
#     t2 = "{:%Y_%m_%d}".format(velocity_interval[ii][1])
#     ####################################################################################modificare nome grafico coon parametri
#     graph_results_date= join( graph_results, "graph_soglia35" + t1 +"__"+ t2 + ".png")
#     stampa(img_SAR,30,5,1, graph_results_date, x, y, vx, vy)


#         #print('\t\t\tUploading results to storage...')
#         #upload_blob(bucket_name, graph_results_date, area.get_glacier_name() 


#         #azi = np.pi*azi/180.0
#         #vx  = np.cos(azi) 
#         #vx = vx*vel   

#         #vy = -np.sin(azi)
#         #vy = vy*vel


#        
#         fig = ff.create_quiver(x,y,vx,vy, scale= 2.0, marker=dict(size='16', 
#             color = mod, #set color equal to a variable
#             colorscale='Viridis',
#             showscale=True))

#         pl.offline.plot(fig, filename=area.get_glacier_name()+"_"+str(velocity_interval[ii][0].date())+"_"+str(velocity_interval[ii][1].date())+'_plot.html')    



#         		#date1obj = datetime.strptime(date1, '%Y%m%dT%H%M%S')
        



         
               
                                    
    
 
