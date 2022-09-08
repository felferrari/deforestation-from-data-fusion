import ee
import time
from libtiff import TIFF
import json
import numpy as np
import os
from skimage.util import view_as_windows, crop
from sklearn.metrics import confusion_matrix
import math as m
import sys


def export_opt_rgb(img, roi, name):
    task_config = {
        'scale': 50, 
        #'crs': crs['crs'],
        'crs':'EPSG:4674',
        'folder':'GEE_rgb_opt',
        #'crsTransform': crs['transform'],
        'fileFormat': 'GeoTIFF',
        'region': roi,
        
    }
    task = ee.batch.Export.image(img.select([
         'B4', 
         'B3',
         'B2',
        ]), f'{name}_rgb', task_config)

    task.start()
    return task

def export_sar_rgb(img, roi, name):
    task_config = {
        'scale': 50, 
        #'crs': crs['crs'],
        'crs':'EPSG:4674',
        'folder':'GEE_rgb_sar',
        #'crsTransform': crs['transform'],
        'fileFormat': 'GeoTIFF',
        'region': roi,
        
    }
    task = ee.batch.Export.image(img.select([
         'VV' 
        ]), f'{name}_rgb', task_config)

    task.start()
    return task

def export_cloud_map(img, roi, name):
    task_config = {
        'scale': 10, 
        #'crs': crs['crs'],
        'crs':'EPSG:4674',
        'folder':'GEE_cmap',
        #'crsTransform': crs['transform'],
        'fileFormat': 'GeoTIFF',
        'region': roi,
    }
    task = ee.batch.Export.image(img.select([
         'MSK_CLDPRB'
        ]), f'{name}_cloud_map', task_config)
    task.start()
    return task
        
def export_opt_2a(img, roi, name):
    task_config = {
        'scale': 10, 
        #'crs': crs['crs'],
        'crs':'EPSG:4674',
        'folder':'GEE_imgs_opt',
        #'crsTransform': crs['transform'],
        'fileFormat': 'GeoTIFF',
        'region': roi,
        
    }
    task = ee.batch.Export.image(img.select([
        'B1', 
        'B2',
        'B3',
        'B4',
        'B5',
        'B6',
        'B7',
        'B8',
        'B8A',
        'B9',
        'B11',
        'B12'
        ]), f'{name}', task_config)

    task.start()
    return task
    
def export_opt_1c(img, roi, name):
    task_config = {
        'scale': 10, 
        #'crs': crs['crs'],
        'crs':'EPSG:4674',
        'folder':'GEE_imgs_opt',
        #'crsTransform': crs['transform'],
        'fileFormat': 'GeoTIFF',
        'region': roi,
        
    }
    task = ee.batch.Export.image(img.select([
        'B1', 
        'B2',
        'B3',
        'B4',
        'B5',
        'B6',
        'B7',
        'B8',
        'B8A',
        'B9',
        'B10',
        'B11',
        'B12'
        ]), f'{name}', task_config)

    task.start()
    return task
        
def export_sar(img, roi, name):
    task_config = {
        'scale': 10, 
        #'crs': crs['crs'],
        'crs':'EPSG:4674',
        'folder':'GEE_imgs_sar',
        #'crsTransform': crs['transform'],
        'fileFormat': 'GeoTIFF',
        'region': roi,
        
    }
    task = ee.batch.Export.image(img.select([
        'VV', 
        'VH'
        ]), f'{name}', task_config)

    task.start()
    return task
    
def export_edge(img, roi, name):  
    task_config = {
        'scale': 10, 
        #'crs': crs['crs'],
        'crs':'EPSG:4674',
        'folder':'GEE_edge',
        #'crsTransform': crs['transform'],
        'fileFormat': 'GeoTIFF',
        'region': roi,
        
    }
    task = ee.batch.Export.image(img.select([
        'MSK_CLDPRB'
        ]), f'{name}', task_config)
    
    task.start()
    return task

def export_cirrus(img, roi, name):  
    task_config = {
        'scale': 10, 
        #'crs': crs['crs'],
        'crs':'EPSG:4674',
        'folder':'GEE_cirrus',
        #'crsTransform': crs['transform'],
        'fileFormat': 'GeoTIFF',
        'region': roi,
        
    }
    task = ee.batch.Export.image(img.select([
        'AOT'
        ]), f'{name}', task_config)
    
    task.start()
    return task

def export_cloud_scl(img, roi, name):  
    task_config = {
        'scale': 10, 
        #'crs': crs['crs'],
        'crs':'EPSG:4674',
        'folder':'GEE_scl',
        #'crsTransform': crs['transform'],
        'fileFormat': 'GeoTIFF',
        'region': roi,
        
    }
    task = ee.batch.Export.image(img.select([
        'SCL',
        'MSK_CLDPRB'
        ]), f'{name}', task_config)
    
    task.start()
    return task
 
