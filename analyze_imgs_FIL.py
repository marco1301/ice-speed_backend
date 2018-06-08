#!/usr/bin/python

import sys
import os
import io
import time
from os.path import join
from datetime import datetime
from google.cloud import storage
from functions.storage import upload_blob, download_blob


#Off-Set tracking library
import velocity.off_set_tracking as off_set_tracking


# Functions definition -
def file_exists(filename):
    bfile = bucket.get_blob(filename)
    if bfile is None:
        return False
    else:
        return True

# def plot_function(filedata):
#     import numpy as np
#     import plotly as pl
#     import plotly.figure_factory as ff 

#     data = np.genfromtxt(filedata, skip_header=1)          
#     x = data[:,0]   
#     y = data[:,1]   
#     vx = data[:,2]      
#     vy = data[:,3] 
#     mod = data[:,4]

#     fig = ff.create_quiver(x,y,vx,vy, scale= 1.0, marker=dict(size='16',
#     color = mod, #set color equal to a variable
#     colorscale='Viridis',
#     showscale=True))

#     pl.offline.plot(fig, filename='testplot.html')    

#     xx = np.linspace(min(x),max(x),int((max(x)-min(x))/200)+1)
#     yy = np.linspace(min(y),max(y),int((max(y)-min(y))/200)+1)

#     fig2 = ff.create_streamline(xx, yy, vx, vy, arrow_scale=.1)

#     pl.offline.plot(fig2, filename='velocity_plot.html')

TESTING = False


if TESTING == True: 
    result_dir = 'RESULTS_'+area.get_param_string()
else:
    result_dir = 'RESULTS'


# Google Storage connection
bucket_name = 'icespeed'
storage_client = storage.Client()
bucket = storage_client.get_bucket(bucket_name)


import areas_functions as af

print "Off-Set tracking import complete"

areas = af.Areas_from_txt('./AREAS/database.txt')

passings = ['ASCENDING', 'DESCENDING'] 

for area in areas:

    # Intervallo limite per l'analisi
    min_days = area.get_min_delta_time()
    max_days = area.get_max_delta_time()

    print('\nArea: '+area.get_glacier_name())

    for passing in passings:
    
        basedir = join(area.get_glacier_name(), passing, result_dir)
    
        blobs = bucket.list_blobs(prefix=join(area.get_glacier_name(), passing, 'S'), delimiter=None)

        list = []
        for blob in blobs:
            list.append(blob.name.split("/")[-1])

	sorted_list = sorted(list, key=lambda x:x.split('_')[4])

        print('\n\tPassing: '+passing+'\n')
        print('\t\tNumero di immagini: '+str(len(sorted_list)) + '\n')
    
        for i in xrange(len(sorted_list)):
            if i != len(sorted_list)-1:

                img1 = sorted_list[i]
                dirname = img1[:-4]
                date1 = img1.split('_')[4]
                date1obj = datetime.strptime(date1, '%Y%m%dT%H%M%S')
                print('\t\t' + str(i) + " - " + img1)
                
                for j in xrange(i+1, len(sorted_list)):
            
                    img2 = sorted_list[j]
                    date2 = img2.split('_')[4]
                    date2obj = datetime.strptime(date2, '%Y%m%dT%H%M%S')
                    delta = date2obj - date1obj
                    image_results_file = 'IMG-' + img2[:-4] + '.txt'
                    terrain_results_file = 'TER-' + img2[:-4] + '.txt'
                    graph_results_file = 'graph-' + img2[:-4] + '.png'
                    terrain_filtered_results_file = 'TER-' + img2[:-4] +'filtered'+'.txt'
                    
                    if  delta.days < max_days and delta.days > min_days and abs(date1obj.minute - date2obj.minute) < 1.0 and (
                            file_exists(join(basedir, dirname, image_results_file))==False or \
                            file_exists(join(basedir, dirname, terrain_results_file))==False
                        ):  #controllo orbite parallele da verificare
                        

                        print ('\t\t\t' + str(j) + ' - Analize with: ' + img2 + ' - days: ' + str(delta.days))

                        print('\t\t\tDownloading images...')
                        if not os.path.exists(join('velocity/data', img1)):
                            download_blob(bucket_name, join(area.get_glacier_name(), passing, img1), join('velocity/data', img1))
                        if not os.path.exists(join('velocity/data', img2)):
                            download_blob(bucket_name, join(area.get_glacier_name(), passing, img2), join('velocity/data', img2))


                        # Run analyze script
                        print('\t\t\tStarting images analyzing script...')
                        #os.system('python velocity/dev_engine_esempio_rafel.py ' + img1 + ' ' + img2)

                        track = off_set_tracking.offset_tracking("velocity/data/"+img1, "velocity/data/"+img2,area, True)
                        try:
                            track.refine_coregistration()
                        except:
                            print "no refinement "
                        
                        track.analyse_pair()
                        track.print_graph()
                        
                        print('\t\t\tUploading results to storage...')
                        upload_blob(bucket_name, 
                                join('./velocity/data/', image_results_file), 
                                join(area.get_glacier_name(), passing, result_dir, dirname, image_results_file))
                        upload_blob(bucket_name, 
                                join('./velocity/data/', terrain_results_file), 
                                join(area.get_glacier_name(), passing, result_dir, dirname, terrain_results_file))
                        upload_blob(bucket_name, 
                                join('./velocity/data/', graph_results_file), 
                                join(area.get_glacier_name(), passing, result_dir, dirname, graph_results_file))
                        upload_blob(bucket_name,
                                join('./velocity/data_filtered/', terrain_filtered_results_file), 
                                join(area.get_glacier_name(), passing, result_dir, dirname, terrain_filtered_results_file)) 

                        # Clean result files
                        os.remove(join('velocity/data/', image_results_file))
                        os.remove(join('velocity/data/', terrain_results_file))
                        os.remove(join('velocity/data/', graph_results_file))
                        os.remove(join('velocity/data_filtered/', terrain_filtered_results_file))
                # END j for loop

        # END i for loop

        # Cancello immagini in locale
        for file in os.listdir('velocity/data/'):
            if file[-4:] == '.tif':
                os.remove(join('velocity/data', file))
    
    # END passings for loop


# END areas for loop

print('\nEND\n\n')
