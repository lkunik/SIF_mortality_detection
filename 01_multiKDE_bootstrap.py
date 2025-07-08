#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Lewis Kunik - University of Utah
# Contact: lewis.kunik@utah.edu
#

#########################################
#### 01_multiKDE_bootstrap.py
#
#

#########################################


#########################################
# Load packages
#########################################

#%%
# File system packages
import os  # operating system library
import sys

# import importlib
# importlib.reload(pixel_classes)
# importlib.reload(functions)
# identify the include directory and add it to the system path
sys.path.append('include')
import pixel_classes
import bootstrap_functions as functions

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
from scipy.stats import gaussian_kde

# time and date packages
import time
from datetime import datetime as dt  # date time library
from datetime import timedelta

# plotting packages
from matplotlib import pyplot as plt  # primary plotting module
import cartopy.crs as ccrs  # spatial plotting library 
from IPython import display


#########################################
# Define Global Filepaths
#########################################
#%%
# directories
dat_dir = 'data/'
dat_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/PhD/chapter2/for_github/data'

# path to processed TROPOMI SIF file
SIF_file = os.path.join(dat_dir, 'TROPOMI_SIF740nm-compSTD-v1.005deg_regrid.16d.2018-05-01_2024-03-31_WesternUS.nc')

# path to processed NASA SRTM DEM file
elev_file = os.path.join(dat_dir, 'SRTM_DEM_05d_WUS.nc')

# path to processed NLCD Tree Canopy Cover file, snapshot of year 2021
TCC_file = os.path.join(dat_dir, 'NLCD_TCC_05d_WUS_2021.nc')

########################################
# Define Global Variables and constants
#########################################

### Variables specific to the dataset 
dat_var = 'sifdc' # this is the name of the variable to be accessed in the dataset
var_units = 'mW/m2/sr/nm'

# Number of bootstrapping iterations to loop through
num_repeats = 500
plot_pixel_maps = True

mortality_detection_year = 2023
include_months_lab = 'Aug-Oct'
include_months_descr = 'late_summer'
discard_months = [11, 12, 1, 2, 3, 4, 5, 6, 7] # define months to discard (months when snowcover may affect signal or when understory may impose strong bias)

control_sample_size = 100


########## Filepaths to mortality/control shapefiles:

estimated_mortality_year = mortality_detection_year - 1

control_polygon_filepath = os.path.join(dat_dir,'shp/control_pixels_05d.shp') # For Bark Beetle Mortality
mortality_polygon_filepath =  os.path.join(dat_dir, 'shp/mortality_pixels_2023_ModerateSevere_MA_25-35p_05d.shp') # For Bark Beetle Mortality
wildfire_studyarea_polygon_filepath = os.path.join(dat_dir, 'shp/EPA_L3_ecoregion_studyarea_wildfire.shp') # For Wildfire Mortality
barkbeetle_studyarea_polygon_filepath = os.path.join(dat_dir, 'shp/EPA_L3_ecoregion_studyarea_beetle.shp') # For Bark Beetle Mortality

# Define parameters for Kernel Density Estimation of land surface characteristic distributions
elevation_bin = 100 # meters
elevation_range = np.linspace(0, 4000, 1000)#[:, None]  #uncomment the [:, None] if using 1D KDE
LAI_bin = 0.25 # unitless
LAI_range = np.linspace(0, 8, 1000)#[:, None] #uncomment the [:, None] if using 1D KDE
TCC_bin = 5 # percent
TCC_range = np.linspace(0, 100, 1000)#[:, None] #uncomment the [:, None] if using 1D KDE

# select for only these years when working with data files
analysis_years = np.array(range(2015, 2024))

# Colors for plotting
mortality_color = "#aa021e" # Line/point color for plots
control_color = "#0219aa"

#########################################
# Define Global Functions
#########################################
# See "include/bootstrap_functions.py" for the functions used in this script

#%%
#########################################
# Begin Main
#########################################

### Load geopandas shapefile data for mortality and control pixel superset

# all mortality pixels 
mortality_polygon_gdf = gpd.read_file(mortality_polygon_filepath)
mortality_polygon_geom = mortality_polygon_gdf.geometry.apply(mapping)
num_mortality_pixels = len(mortality_polygon_gdf)

# Get the bounding box (min/max lon/lat) of the mortality polygons
minx, miny, maxx, maxy = mortality_polygon_gdf.total_bounds
print(f"Mortality pixel extent: lon [{minx}, {maxx}], lat [{miny}, {maxy}]")

# all control pixels
control_polygon_gdf = gpd.read_file(control_polygon_filepath)
control_polygon_geom = control_polygon_gdf.geometry.apply(mapping)
num_control_pixels = len(control_polygon_geom)

# Get the bounding box (min/max lon/lat) of the control polygons
minx, miny, maxx, maxy = control_polygon_gdf.total_bounds
print(f"Control pixel extent: lon [{minx}, {maxx}], lat [{miny}, {maxy}]")

# Plot map of the control and mortality pixels
control_centroids = control_polygon_gdf.centroid
mortality_centroids = mortality_polygon_gdf.centroid
functions.plot_mortality_control_pixel_map(control_centroids, mortality_centroids)

#%%

#########################################
# Load DEM, Tree Canopy Cover and satellite RS datasets
#########################################

# Load NASA SRTM elevation (pre-gridded to 0.05°)
print('Loading elevation dataset')
SRTM_xr = xr.open_dataset(elev_file).squeeze('band').rename({'lon':'x', 'lat':'y'}).sortby(['x', 'y'])
SRTM_xr.rio.write_crs(4326, inplace = True) # specify WGS84 as CRS

# Load NLCD Tree Canopy Cover (pre-gridded to 0.05°)
TCC_xr = xr.open_dataset(TCC_file).squeeze('band').rename({'lon':'x', 'lat':'y'}).sortby(['x', 'y']) # Assume grid matches with elevation dataset (already checked)
TCC_xr.rio.write_crs(4326, inplace = True) # specify WGS84 as CRS

# Load TROPOMI SIF 0.05° gridded dataset
SIF_xr = xr.open_dataset(SIF_file, decode_coords='all').sortby(['lon', 'lat'])
SIF_xr = SIF_xr.sel(time=SIF_xr['time.year'].isin(analysis_years))
SIF_xr.rio.write_crs(4326, inplace=True)


#%%
##################################################################################
# calculate the distributions of elevation and tree canopy cover across the mortality pixels
##################################################################################
   
SRTM_mortality_kde, SRTM_mortality_density, SRTM_mortality_non_nan = functions.compute_kde_density_from_gdf(mortality_polygon_gdf, SRTM_xr, 'elevation', elevation_range)
SRTM_control_kde, SRTM_control_density, SRTM_control_non_nan = functions.compute_kde_density_from_gdf(control_polygon_gdf, SRTM_xr, 'elevation', elevation_range)
TCC_mortality_kde, TCC_mortality_density, TCC_mortality_non_nan = functions.compute_kde_density_from_gdf(mortality_polygon_gdf, TCC_xr, 'tree_canopy_cover', TCC_range)
TCC_control_kde, TCC_control_density, TCC_control_non_nan = functions.compute_kde_density_from_gdf(control_polygon_gdf, TCC_xr, 'tree_canopy_cover', TCC_range)

# get mean for later scaling
elev_control_mean = np.mean(SRTM_control_non_nan)
TCC_control_mean = np.mean(TCC_control_non_nan)

# clip the SRTM and TCC datasets to the control polygons
SRTM_xr_control_clip = SRTM_xr.rio.clip(control_polygon_geom, crs=4326)
TCC_xr_control_clip = TCC_xr.rio.clip(control_polygon_geom, crs=4326)

# Flatten the control datasets for use in the bootstrap step
SRTM_control_flat = SRTM_xr_control_clip.elevation.data.flatten()
TCC_control_flat = TCC_xr_control_clip.tree_canopy_cover.data.flatten()

#%%
# Set global matplotlib font sizes for axes and legends
plt.rcParams.update({
   'axes.labelsize': 16,
   'axes.titlesize': 18,
   'xtick.labelsize': 14,
   'ytick.labelsize': 14,
   'legend.fontsize': 14
})
# Plot histograms of SRTM_mortality_non_nan and SRTM_control_non_nan with KDE overlays
plt.figure(figsize=(12, 5))

# Elevation histogram + KDE for Mortality
plt.subplot(1, 2, 1)
plt.hist(SRTM_mortality_non_nan, bins=30, color=mortality_color, alpha=0.7, density=True, label='Mortality Histogram')
plt.plot(elevation_range, SRTM_mortality_density, color='black', linewidth=2, label='Mortality KDE')
plt.xlabel('Elevation (m)')
plt.ylabel('Density')
plt.title('Mortality Pixel Elevation')
plt.legend()

# Elevation histogram + KDE for Control
plt.subplot(1, 2, 2)
plt.hist(SRTM_control_non_nan, bins=30, color=control_color, alpha=0.7, density=True, label='Control Histogram')
plt.plot(elevation_range, SRTM_control_density, color='black', linewidth=2, label='Control KDE')
plt.xlabel('Elevation (m)')
plt.ylabel('Density')
plt.title('Control Pixel Elevation')
plt.legend()

plt.tight_layout()
plt.show()

#%%
# Plot histograms of TCC_mortality_non_nan and TCC_control_non_nan with KDE overlays
plt.figure(figsize=(12, 5))

# Tree Canopy Cover histogram + KDE for Mortality
plt.subplot(1, 2, 1)
plt.hist(TCC_mortality_non_nan, bins=30, color=mortality_color, alpha=0.7, density=True, label='Mortality Histogram')
plt.plot(TCC_range, TCC_mortality_density, color='black', linewidth=2, label='Mortality KDE')
plt.xlabel('Tree Canopy Cover (%)')
plt.ylabel('Density')
plt.title('Mortality Pixel Tree Canopy Cover')
plt.legend()

# Tree Canopy Cover histogram + KDE for Control
plt.subplot(1, 2, 2)
plt.hist(TCC_control_non_nan, bins=30, color=control_color, alpha=0.7, density=True, label='Control Histogram')
plt.plot(TCC_range, TCC_control_density, color='black', linewidth=2, label='Control KDE')
plt.xlabel('Tree Canopy Cover (%)')
plt.ylabel('Density')
plt.title('Control Pixel Tree Canopy Cover')
plt.legend()

plt.tight_layout()
plt.show()


#%%
   
# Set global matplotlib font sizes for axes and legends
plt.rcParams.update({
   'axes.labelsize': 14,
   'axes.titlesize': 13,
   'xtick.labelsize': 12,
   'ytick.labelsize': 12,
   'legend.fontsize': 12
})

if plot_pixel_maps:

   for ensemble_iter in range(num_repeats):

      # print(f'Bootstrap iteration: {ensemble_iter:02d}')

      #########################################
      # Establish Mortality datastructure
      #########################################

      # Randomly sample rows of the mortality_polygon_gdf object
      mortality_centroids = []
      mortality_polygon_gdf_sampled = mortality_polygon_gdf.sample(num_mortality_pixels, replace=True)

      # Populate Mortality sample with pixels
      for pixel_center in mortality_polygon_gdf_sampled.centroid:
         mortality_centroids.append(gpd.points_from_xy([pixel_center.x], [pixel_center.y])[0])
      mortality_centroids = gpd.GeoSeries(mortality_centroids)


      #########################################
      # Establish Control datastructure
      #########################################

      # Sample new values based on the densities
      sampled_elev_values = SRTM_mortality_kde.resample(control_sample_size).flatten()
      sampled_TCC_values = TCC_mortality_kde.resample(control_sample_size).flatten()

      # Find indices in the control domain where both elevation and LAI match closely
      matching_indices = []
      control_polygon_gdf_KDE_FILTER = gpd.GeoDataFrame(columns=control_polygon_gdf.columns, crs=control_polygon_gdf.crs)

      for elev_val, TCC_val in zip(sampled_elev_values, sampled_TCC_values):
         # Calculate the normalized differences of each var
         elev_diffs = np.abs(SRTM_control_flat - elev_val)/elev_control_mean # divide by mean to normalize
         TCC_diffs = np.abs(TCC_control_flat - TCC_val)/TCC_control_mean # divide by mean to normalize
         
         # Combine constraints: find indices where combined differences are minimal
         overall_diffs = elev_diffs + TCC_diffs
         matching_idx = np.nanargmin(overall_diffs) # identify the index of the minimum difference
         
         # Append the index if not already present (retains only unique pixels)
         if matching_idx not in matching_indices:
            matching_indices.append(matching_idx)

      # Convert the original indices to unique x/y coordinates
      sampled_coords = np.unravel_index(matching_indices, SRTM_xr_control_clip.elevation.shape)

      # Extract x and y coordinates as indices from the original datasets' grid
      sampled_x_icoords = sampled_coords[1]
      sampled_y_icoords = sampled_coords[0]

      # Loop through the sampled coordinates and recreate the gridded points where elevations were selected
      for ii in range(len(sampled_x_icoords)):
         x_coord = SRTM_xr_control_clip.x.values[sampled_x_icoords[ii]]
         y_coord = SRTM_xr_control_clip.y.values[sampled_y_icoords[ii]]

         # Find the closest point in control_polygon_gdf for each x, y coordinate pair
         distances = control_polygon_gdf.geometry.apply(lambda geom: geom.distance(gpd.points_from_xy([x_coord], [y_coord])[0]))
         closest_point_index = distances.idxmin()
         closest_point = control_polygon_gdf.loc[closest_point_index]
         
         # Filter control_polygon_gdf for points that are matched to these sets of coordinates
         control_polygon_gdf_KDE_FILTER = pd.concat([control_polygon_gdf_KDE_FILTER, closest_point.to_frame().T])

      # control_polygon_gdf_KDE_FILTER is now just like control_polygon_gdf but with a similar elevation distrib to mortality pixels
      num_control_pixels = len(control_polygon_gdf_KDE_FILTER)

      # Randomly sample rows of the mortality_polygon_gdf object
      control_polygon_gdf_sampled = control_polygon_gdf_KDE_FILTER.sample(num_control_pixels, replace=True)

      SRTM_control_kde_sampled, SRTM_control_density_sampled, SRTM_control_non_nan_sampled = functions.compute_kde_density_from_gdf(control_polygon_gdf_sampled, SRTM_xr, 'elevation', elevation_range)
      TCC_control_kde_sampled, TCC_control_density_sampled, TCC_control_non_nan_sampled = functions.compute_kde_density_from_gdf(control_polygon_gdf_sampled, TCC_xr, 'tree_canopy_cover', TCC_range)

      # Populate Control sample with pixels
      control_centroids = []
      for pixel_center in control_polygon_gdf_sampled.centroid:
         control_centroids.append(gpd.points_from_xy([pixel_center.x], [pixel_center.y])[0])
      control_centroids = gpd.GeoSeries(control_centroids)



      # Clear the previous output
      display.clear_output(wait=True)
      plt.figure(figsize=(12, 10))
      ax_map = plt.subplot2grid((2, 2), (0, 0), rowspan=2, projection=ccrs.PlateCarree())
      ax_tcc_kde = plt.subplot2grid((2, 2), (0, 1))
      ax_elev_kde = plt.subplot2grid((2, 2), (1, 1))


      # Adjust subplot positions to add buffer/margins
      plt.subplots_adjust(left=0.03, right=0.97, top=0.96, bottom=0.02, wspace=0.15, hspace=0.15)
      box_tcc = ax_tcc_kde.get_position()
      box_elev = ax_elev_kde.get_position()
      # Shrink and move the TCC plot
      ax_tcc_kde.set_position([box_tcc.x0 + 0.04, box_tcc.y0 + 0.04, box_tcc.width * 0.9, box_tcc.height * 0.9])
      # Shrink and move the Elevation plot
      ax_elev_kde.set_position([box_elev.x0 + 0.04, box_elev.y0 + 0.04, box_elev.width * 0.9, box_elev.height * 0.9])
      # Move ax_map to align its top with the top of the figure, keeping its horizontal position
      box_map = ax_map.get_position()
      new_height = box_map.height + (1.0 - box_map.y1)
      ax_map.set_position([box_map.x0, 1.0 - new_height, box_map.width, new_height])

      # TCC comparison plot (top right)
      ax_tcc_kde.hist(TCC_control_non_nan_sampled, bins=30, color=control_color, alpha=0.5, 
                  density=True, label='Control\n(Sampled)')
      ax_tcc_kde.plot(TCC_range, TCC_control_density, color=control_color,
                  linewidth=2, linestyle='--', label='Control KDE\n(Original)')
      ax_tcc_kde.plot(TCC_range, TCC_control_density_sampled, color=control_color, 
                  linewidth=2, label='Control KDE\n(Sampled)')
      ax_tcc_kde.plot(TCC_range, TCC_mortality_density, color=mortality_color, 
                  linewidth=2, label='Mortality KDE')
      
      ax_tcc_kde.set_xlabel('Tree Canopy Cover (%)')
      ax_tcc_kde.set_ylabel('Density')
      ax_tcc_kde.set_title('Tree Canopy Cover Distribution')
      ax_tcc_kde.legend()

      # Elevation comparison plot (bottom right)
      ax_elev_kde.hist(SRTM_control_non_nan_sampled, bins=30, color=control_color, alpha=0.5, 
                  density=True, label='Control\n(Sampled)')
      ax_elev_kde.plot(elevation_range, SRTM_control_density, color=control_color, 
                  linewidth=2, linestyle='--', label='Control KDE\n(Original)')
      ax_elev_kde.plot(elevation_range, SRTM_control_density_sampled, color=control_color, 
                  linewidth=2, label='Control KDE\n(Sampled)')
      ax_elev_kde.plot(elevation_range, SRTM_mortality_density, color=mortality_color, 
                  linewidth=2, label='Mortality KDE')


      ax_elev_kde.set_xlabel('Elevation (m)')
      ax_elev_kde.set_ylabel('Density')
      ax_elev_kde.set_title('Elevation Distribution')
      ax_elev_kde.legend()

      functions.plot_mortality_control_pixel_map(control_centroids, mortality_centroids, plotTitle=f'Bootstrap Iteration: {ensemble_iter:02d}', fig=plt.gcf(), ax=ax_map)

      # plt.tight_layout()
      display.display(plt.gcf())
      # time.sleep(0.1)
      plt.close()
      


# %%
#########################################
# Begin Bootstrapping
#########################################

summary_ann_mean_arr = []
cumulative_summary_ann_mean_arr = []
summary_1yr_delta_arr = []
cumulative_summary_1yr_delta_arr = []

# Set global matplotlib font sizes for axes and legends
plt.rcParams.update({
   'axes.labelsize': 11,
   'axes.titlesize': 12,
   'xtick.labelsize': 11,
   'ytick.labelsize': 11,
   'legend.fontsize': 11
})


for ensemble_iter in range(num_repeats):

   print(f'Bootstrap iteration: {ensemble_iter:02d}')

   #########################################
   # Establish Mortality datastructure
   #########################################

   # Randomly sample rows of the mortality_polygon_gdf object
   mortality_polygon_gdf_sampled = mortality_polygon_gdf.sample(num_mortality_pixels, replace=True)

   # set up disturbed PixelGroup
   mortality_sample = pixel_classes.PixelGroup()

   # Populate Mortality sample with pixels
   for pixel_center in mortality_polygon_gdf_sampled.centroid:
      pixel = pixel_classes.Pixel(center_lon = pixel_center.x,
                                    center_lat = pixel_center.y,
                                    deg_res = 0.05)
      mortality_sample.add_pixel(pixel)

   #########################################
   # Establish Control datastructure
   #########################################

   # Sample new values based on the densities
   sampled_elev_values = SRTM_mortality_kde.resample(control_sample_size).flatten()
   sampled_TCC_values = TCC_mortality_kde.resample(control_sample_size).flatten()

   # Find indices in the control domain where both elevation and LAI match closely
   matching_indices = []
   control_polygon_gdf_KDE_FILTER = gpd.GeoDataFrame(columns=control_polygon_gdf.columns, crs=control_polygon_gdf.crs)

   for elev_val, TCC_val in zip(sampled_elev_values, sampled_TCC_values):
      # Calculate the normalized differences of each var
      elev_diffs = np.abs(SRTM_control_flat - elev_val)/elev_control_mean # divide by mean to normalize
      TCC_diffs = np.abs(TCC_control_flat - TCC_val)/TCC_control_mean # divide by mean to normalize
      
      # Combine constraints: find indices where combined differences are minimal
      overall_diffs = elev_diffs + TCC_diffs
      matching_idx = np.nanargmin(overall_diffs) # identify the index of the minimum difference
      
      # Append the index if not already present (retains only unique pixels)
      if matching_idx not in matching_indices:
         matching_indices.append(matching_idx)

   # Convert the original indices to unique x/y coordinates
   sampled_coords = np.unravel_index(matching_indices, SRTM_xr_control_clip.elevation.shape)

   # Extract x and y coordinates as indices from the original datasets' grid
   sampled_x_icoords = sampled_coords[1]
   sampled_y_icoords = sampled_coords[0]

   # Loop through the sampled coordinates and recreate the gridded points where elevations were selected
   for ii in range(len(sampled_x_icoords)):
      x_coord = SRTM_xr_control_clip.x.values[sampled_x_icoords[ii]]
      y_coord = SRTM_xr_control_clip.y.values[sampled_y_icoords[ii]]

      # Find the closest point in control_polygon_gdf for each x, y coordinate pair
      distances = control_polygon_gdf.geometry.apply(lambda geom: geom.distance(gpd.points_from_xy([x_coord], [y_coord])[0]))
      closest_point_index = distances.idxmin()
      closest_point = control_polygon_gdf.loc[closest_point_index]
      
      # Filter control_polygon_gdf for points that are matched to these sets of coordinates
      control_polygon_gdf_KDE_FILTER = pd.concat([control_polygon_gdf_KDE_FILTER, closest_point.to_frame().T])

   # control_polygon_gdf_KDE_FILTER is now just like control_polygon_gdf but with a similar elevation distrib to mortality pixels
   num_control_pixels = len(control_polygon_gdf_KDE_FILTER)

   # Randomly sample rows of the mortality_polygon_gdf object
   control_polygon_gdf_sampled = control_polygon_gdf_KDE_FILTER.sample(num_control_pixels, replace=True)

   SRTM_control_kde_sampled, SRTM_control_density_sampled, SRTM_control_non_nan_sampled = functions.compute_kde_density_from_gdf(control_polygon_gdf_sampled, SRTM_xr, 'elevation', elevation_range)
   TCC_control_kde_sampled, TCC_control_density_sampled, TCC_control_non_nan_sampled = functions.compute_kde_density_from_gdf(control_polygon_gdf_sampled, TCC_xr, 'tree_canopy_cover', TCC_range)


   # set up Pixel Group for control
   control_sample = pixel_classes.PixelGroup()

   # Populate Control sample with pixels
   for pixel_center in control_polygon_gdf_sampled.centroid:
      pixel = pixel_classes.Pixel(center_lon = pixel_center.x,
                                    center_lat = pixel_center.y,
                                    deg_res = 0.05)
      control_sample.add_pixel(pixel)


   #######################################################
   # Add data to the Mortality and Control pixel group datastructures
   #######################################################

   mortality_sample.add_xr_to_all_pixels(xr_da = SIF_xr[dat_var], var_name = dat_var, 
                                       units = var_units)

   control_sample.add_xr_to_all_pixels(xr_da = SIF_xr[dat_var], var_name = dat_var, 
                                       units = var_units)
   
   ### Trim away winter months
   mortality_sample.filter_var_by_months(varname_to_filter = dat_var,
                                          months_to_remove = discard_months)
   control_sample.filter_var_by_months(varname_to_filter = dat_var,
                                          months_to_remove = discard_months)
   
   ### Summarize by year
   mortality_sample.summarize_annual(var_name = dat_var)
   control_sample.summarize_annual(var_name = dat_var)
   
   ### Compute mean statistics (mortality mean/std, control mean/std, mean diff, p-values of Mann-Whitney U and Kolmogorov-Smirnov tests)
   summary_ann_mean_df = functions.stat_summary(mortality_sample, control_sample, dat_var)
   summary_1yr_delta_df = functions.stat_summary(mortality_sample, control_sample, dat_var, shift_years = 1)

   summary_ann_mean_arr.append(summary_ann_mean_df)
   summary_1yr_delta_arr.append(summary_1yr_delta_df)



   if plot_pixel_maps:
      summary_ann_mean_cumulative = functions.calculate_mean_across_dataframes(summary_ann_mean_arr)
      summary_ann_mean_cumulative = functions.add_significance_columns(summary_ann_mean_cumulative)
      summary_ann_mean_cumulative['num_repetitions'] = ensemble_iter + 1
      summary_1yr_delta_cumulative = functions.calculate_mean_across_dataframes(summary_1yr_delta_arr)
      summary_1yr_delta_cumulative = functions.add_significance_columns(summary_1yr_delta_cumulative)
      summary_1yr_delta_cumulative['num_repetitions'] = ensemble_iter + 1

      cumulative_summary_ann_mean_arr.append(summary_ann_mean_cumulative)
      cumulative_summary_1yr_delta_arr.append(summary_1yr_delta_cumulative)

      if ensemble_iter == 0:
         continue

      display.clear_output(wait=True)
      fig_boot = plt.figure(figsize=(10, 7))
      ax_annual = plt.subplot2grid((3, 2), (0, 0), fig = fig_boot)
      ax_1yrdelta = plt.subplot2grid((3, 2), (0, 1), fig = fig_boot)
      ax_grpDiffCI = plt.subplot2grid((3, 2), (1, 0), rowspan=2, fig = fig_boot)
      ax_MannWhitney = plt.subplot2grid((3, 2), (1, 1), rowspan=2, fig = fig_boot)




      functions.plot_summary(summary_df = summary_ann_mean_df, 
            plot_title = f'Bootstrap iter={ensemble_iter:02d} - Annual Mean', 
            yax_label = r'SIF$_{dc}$ (mW m$^{-2}$ sr$^{-1}$ nm$^{-1}$)',
            mortality_year = estimated_mortality_year,
            fig=fig_boot, ax = ax_annual)
      
      # time.sleep(0.1)
      
      functions.plot_summary(summary_df = summary_1yr_delta_df, 
            plot_title = f'Bootstrap iter={ensemble_iter:02d} - 1yr $\Delta$SIF', 
            yax_label = r'1yr $\Delta$SIF$_{dc}$'+'\n' + r'(mW m$^{-2}$ sr$^{-1}$ nm$^{-1}$)',
            mortality_year = estimated_mortality_year,
            fig=fig_boot, ax = ax_1yrdelta, draw_box = True)

      functions.plot_bootstrap_CI_stabilization(cumulative_df_arr = cumulative_summary_1yr_delta_arr,
            yyyy = estimated_mortality_year, yax_label = '1yr $\Delta$SIF, Mortality minus Control\n'+r'(mW m$^{-2}$ sr$^{-1}$ nm$^{-1}$)', fig=fig_boot, ax = ax_grpDiffCI)
      functions.plot_bootstrap_pval_stabilization(cumulative_df_arr = cumulative_summary_1yr_delta_arr,
            yyyy = estimated_mortality_year, fig=fig_boot, ax = ax_MannWhitney)

      plt.tight_layout()
      display.display(fig_boot)
      time.sleep(0.1)
      plt.close()  
# %%

summary_ann_mean = functions.calculate_mean_across_dataframes(summary_ann_mean_arr)
summary_ann_mean = functions.add_significance_columns(summary_ann_mean)
summary_1yr_delta = functions.calculate_mean_across_dataframes(summary_1yr_delta_arr)
summary_1yr_delta = functions.add_significance_columns(summary_1yr_delta)

fig, ax = plt.subplots(figsize=(5, 3))
functions.plot_summary(summary_df = summary_ann_mean, 
      plot_title = r'Bootstrap summary - Annual Mean', 
      yax_label = r'SIF$_{dc}$ (mW m$^{-2}$ sr$^{-1}$ nm$^{-1}$)',
      mortality_year = estimated_mortality_year,
      fig=fig, ax=ax, plot_significance_chars=True)
plt.show()


fig, ax = plt.subplots(figsize=(5, 3))
functions.plot_summary(summary_df = summary_1yr_delta, 
      plot_title = r'Bootstrap summary - 1yr $\Delta$SIF', 
      yax_label = r'1yr $\Delta$SIF$_{dc}$' + r'(mW m$^{-2}$ sr$^{-1}$ nm$^{-1}$)',
      mortality_year = estimated_mortality_year,
      fig=fig, ax=ax, plot_significance_chars=True)

plt.show()
# %%
