#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Lewis Kunik - University of Utah
# Contact: lewis.kunik@utah.edu



#########################################
# Load packages
#########################################

#%%
# File system packages
import os  # operating system library
import sys

# runtime packages
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# Suppress UserWarning about centroid in geographic CRS globally
warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS. Results from 'centroid' are likely incorrect.*")

# numerical/data management packages
import numpy as np
import xarray as xr  # for multi dimensional data
import pandas as pd

# shapefile/geospatial packages
import geopandas as gpd
from shapely.geometry import mapping


# plotting packages
from matplotlib import pyplot as plt  # primary plotting module
import cartopy.crs as ccrs  # spatial plotting library 
from cartopy.io import img_tiles  # cartopy's implementation of webtiles
from IPython import display

#########################################
# Define Global Filepaths
#########################################

# directories
dat_dir = 'data/'

wildfire_studyarea_polygon_filepath = os.path.join(dat_dir, 'shp/EPA_L3_ecoregion_studyarea_wildfire.shp') # For Wildfire Mortality
barkbeetle_studyarea_polygon_filepath = os.path.join(dat_dir, 'shp/EPA_L3_ecoregion_studyarea_beetle.shp') # For Bark Beetle Mortality


########################################
# Define Global Variables and constants
#########################################

mortality_color = "#aa021e" # Line/point color for plots
control_color = "#0219aa"

#########################################
# Set up mapping info
#########################################

map_extent =  [-125, -117.8, 34.6, 43.8] 

# use Google Satellite imagery as basemap
tiler = img_tiles.GoogleTiles(style='satellite')
crs = tiler.crs # set crs of map tiler
alpha = 0.8  # transparency 0-1
transform = ccrs.PlateCarree()  # transform specifies the crs that the data is in
map_tiler_zoom = 7

### Load state boundaries
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature
fn = shpreader.natural_earth(
   resolution='10m', category='cultural', 
   name='admin_1_states_provinces',
)
reader = shpreader.Reader(fn)
states = [x for x in reader.records() if x.attributes["admin"] == "United States of America"] # get all states in US
states_geom = cfeature.ShapelyFeature([x.geometry for x in states], ccrs.PlateCarree())

# Load study area bounds (for map display)
study_area_bounds_gdf = gpd.read_file(barkbeetle_studyarea_polygon_filepath)


def setup_pixel_map(fig = None, ax = None):
   
   if fig is None or ax is None:
      # Set up the figure and axes for the map
      fig, ax = plt.subplots(subplot_kw={'projection': crs}, figsize=(7, 7))
   
   # Set extent of map before adding base image
   ax.set_extent(map_extent, crs=ccrs.PlateCarree())  # PlateCarree is default lat/lon
 
   # add Google satellite imagery
   ax.add_image(tiler, map_tiler_zoom, interpolation='none', alpha=0.7) # Add base image

   # add state boundaries to the map
   ax.add_feature(states_geom, facecolor="none", edgecolor="black", linewidth=1)

   # add study area ecoregion boundary to map
   ax.add_geometries(study_area_bounds_gdf.geometry, crs=transform, facecolor="white", edgecolor="black", linewidth=0.8, zorder=1, alpha = 0.5)

   return fig, ax


def plot_mortality_control_pixel_map(control_centroids, mortality_centroids, markersize = 10, fig = None, ax = None, plotTitle=""):
   
   if fig is None or ax is None:
      fig, ax = setup_pixel_map() # set up blank map
   else:
      fig, ax = setup_pixel_map(fig, ax)

   # Plot centroids of control and mortality pixels
   ax.scatter(control_centroids.x, control_centroids.y, color=control_color, s=markersize, marker='o', transform=ccrs.PlateCarree(), label='Control')
   ax.scatter(mortality_centroids.x, mortality_centroids.y, color=mortality_color, s=markersize, marker='o', transform=ccrs.PlateCarree(), label='Mortality')

   # Plot legend
   ax.legend(loc='upper right', fontsize=10, borderpad=0.5, facecolor='white')
   ax.set_title(plotTitle, fontsize=14)
   # plt.show()



