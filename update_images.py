#!/usr/bin/python

from __future__ import print_function
import ee
import os
#import ee.mapclient
import datetime
#import matplotlib.pyplot as plt
import numpy as np
from ee.batch import Export
import plotly
from plotly.graph_objs import Scatter, Layout
import sys
import time
import difflib
import boto

from google.cloud import storage
from functions.storage import upload_blob

bucket_name = "icespeed"


# Functions definitions
def ee_export_image_to_bucket(image_name, bucket_name, exported_area, area_name, folder):
    # Select image
    print(image_name)
    image = ee.Image("COPERNICUS/S1_GRD/" + image_name)
    # add image equalitation for imporve matching 
    image = ee.Image(image.select(0))
    #ppp = [5,95]
    result = image.select(0).reduceRegion(reducer=ee.Reducer.minMax(),geometry=ee.Geometry.Polygon(exported_area), maxPixels=37000000000)
    min_value = result.getInfo()['VV_min']
    max_value = result.getInfo()['VV_max']
    print(min_value)
    print(max_value)
    image = image.select(0).add(-min_value).multiply(255.0/(max_value-min_value))
    # Creating and run export task
    task = Export.image.toCloudStorage(
        image=image.select(0),
        bucket=bucket_name,
        fileNamePrefix=area_name + "/" + folder + "/" + image_name,
        scale=10,
        region=exported_area,
    )
    task.start()
    
def save_list(list, file):
    thefile = open(file, 'w')
    for item in list:
      thefile.write("%s\n" % item)
    
class EE_Tasks_Handler():
  def update_task_list(self):
    self.tasks = ee.batch.Task.list()
   
  def __init__(self):
    self.update_task_list()
    self.Run, self.Rea, self.Comp = 0,0,0

  def tasks_is_runnig(self):
    control = False 
    for task in self.tasks:
      if task.config['state'] in (u'RUNNING',u'UNSUBMITTED',u'READY'):
        control = True
    return control

  def wait_until_all_tasks_finish(self):
      print('\n\nWaiting running tasks...')
      while self.tasks_is_runnig():
          self.tasks_monitoring()
          time.sleep(10)
          self.update_task_list()

  def cancel_all_running_tasks(self):
    for task in self.tasks:
      if task.config['state'] in (u'RUNNING',u'UNSUBMITTED',u'READY'):
          print('canceling %s' % task)
          task.cancel()
  
  def cancel_completed_tasks(self):
    for task in self.tasks:
      if task.config['state'] in (u'COMPLETED'):
          print('Canceling %s' % task)
          task.cancel()

  def tasks_monitoring(self):
    for task in self.tasks:
      if task.config['state'] == u'RUNNING':
        self.Run += 1 
      if task.config['state'] == u'READY':
        self.Rea += 1
      if task.config['state'] == u'COMPLETED':
        self.Comp += 1
    print('Running task: ' + str(self.Run) + ' - Ready task: ' + str(self.Rea) + ' - Completed task: ' + str(self.Comp), end='\r')
    sys.stdout.flush()
    self.Run, self.Rea, self.Comp = 0,0,0
# end function definition


    
ee.Initialize()
print('\n--------------------------')
print('   Earth Engine Started')
print('--------------------------')

# Control tasks by arguments
if len(sys.argv) > 1:
    Tasks_H1 = EE_Tasks_Handler()
    if sys.argv[1]=="delete":
        print('\nDeleting all running tasks\n')
        Tasks_H1.cancel_all_running_tasks()
    if sys.argv[1]=="monitor":
        print('\nWaiting tasks is running...')
        Tasks_H1.wait_until_all_tasks_finish()
        print('\nTasks completed.\n')
    if sys.argv[1]=="list":
        print('\nListing tasks...')
        Tasks_H1.update_task_list()
    if sys.argv[1]=="clear":
        print('\nCancel completed tasks...')
        Tasks_H1.cancel_completed_tasks()
    sys.exit()


today = time.strftime("%Y-%m-%d")

import areas_functions as af

areas = af.Areas_from_txt('./AREAS/database.txt')

passings = ['DESCENDING', 'ASCENDING']


for area in areas:

    area_name = area.get_glacier_name()
    
    print('\nSelected Area:', area_name)
    
    p1 = ee.Geometry.Point(area.get_corners()[0])
    p2 = ee.Geometry.Point(area.get_corners()[1])
    p3 = ee.Geometry.Point(area.get_corners()[2])
    p4 = ee.Geometry.Point(area.get_corners()[3])

    #print(area, p1, p2, p3 ,p4)
    
    #sys.exit()

    for passing in passings:
        
        print('\n\tSelected passing: '+passing)
        
        sentinel1 = ee.ImageCollection('COPERNICUS/S1_GRD').filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
                                                                .filter(ee.Filter.eq('orbitProperties_pass', passing)) \
                                                                .filterDate('2015-01-01', today) \
                                                                .filterBounds(p1) \
                                                                .filterBounds(p2) \
                                                                .filterBounds(p3) \
                                                                .filterBounds(p4)
                                                                #.filter(ee.Filter.contains({leftValue: geometry3, rightField: ".geo"}))

        N_sentinel = sentinel1.size().getInfo()
        print('\n\t\tAvailable images on EARTH ENGINE since 2015-01-01:', N_sentinel)

        lista_sentinel = sentinel1.toList(100)
        A_lista_sentinel = lista_sentinel.getInfo()
        sentinel_date = []

        for elem in A_lista_sentinel:
            id = elem['id']
            date_i = id.rsplit('_')[5].rsplit('T')[0]
            sentinel_date.append(datetime.datetime.strptime(date_i ,'%Y%m%d'))

        N_img = sentinel1.size().getInfo()
        lista_img = sentinel1.toList(100)
        A_lista_img = lista_img.getInfo()

        # List of selected images con Earth Engine
        lista_img_earth = ""
        for i in xrange(N_img):
            image = ee.Image(lista_img.get(i))
            img_name = A_lista_img[i]['id'].encode('ascii','ignore').rsplit('/')[-1]+".tif"
            lista_img_earth = lista_img_earth + "\n" + img_name
        list_img_earth_lines = lista_img_earth.splitlines()
        list_img_earth_lines.remove("")
        #save_list(list_img_earth_lines, 'list_earth.txt')
            
        # List of images saved on Google Cloud Storage
        list_img_cloud_lines = []
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=area_name + '/' + passing + '/S', delimiter=None)
        for blob in blobs:
            list_img_cloud_lines.append(blob.name.split("/")[-1])
        print('\n\t\tImages exported on Google Storage: ' + str(len(list_img_cloud_lines)))
        #save_list(list_img_cloud_lines, 'list_cloud_new.txt')

        # List and number of images not saved on GCS (diff)
        d = difflib.Differ()
        diff=list(d.compare(list_img_cloud_lines, list_img_earth_lines))
        diff_num = len([ i for i, word in enumerate(diff) if word.startswith('+ ') ])

        # If there are images to export to GCS
        if diff_num > 0:
            export = 'Y'
            print('\n\t\t' + str(diff_num) + ' image[s] not on Google Cloud Storage')
            print('\n\t\tExporting following image[s] to GCS:')
            for elem in diff:
                if elem[:1]=="+" and elem[2:]!="":
                    img_name = elem[2:].replace(".tif","")
                    print('\t\t\t' + img_name)
                    # Export image to GCS
                    ee_export_image_to_bucket(img_name, bucket_name, area.corners, area_name, passing)       
        else:
            print('\n\t\tAll images exported to Google Cloud Storage')
        
    # end for loop passings
    
# end for loop areas

# Monitoring exporting tasks
if 'export' in locals():
    Tasks_H1 = EE_Tasks_Handler()
    Tasks_H1.wait_until_all_tasks_finish()
    print('\nExport completed.\n')
else:
    print('\nEND\n')
    


# plotly.offline.plot({
#   "data": [Scatter(x= sentinel_date, y= np.arange(len(sentinel_date)),  mode = 'markers' , name = 'ASC'), Scatter(x= desc_date, y= np.arange(len(desc_date)),  mode = 'markers', name = 'DESC' )],
#     "layout": Layout(title="Sentinel-1 images acquired over AOI", xaxis = dict(
#         title= 'Pop'))
#     })
