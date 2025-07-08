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

# identify the include directory and add it to the system path
sys.path.append('include')
import pixel_classes

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
from cartopy.io import img_tiles  # cartopy's implementation of webtiles
from IPython import display

#########################################
# Define Global Filepaths
#########################################

# directories
dat_dir = 'data/'
dat_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/bootstrap/code/for_github/data'

wildfire_studyarea_polygon_filepath = os.path.join(dat_dir, 'shp/EPA_L3_ecoregion_studyarea_wildfire.shp') # For Wildfire Mortality
barkbeetle_studyarea_polygon_filepath = os.path.join(dat_dir, 'shp/EPA_L3_ecoregion_studyarea_beetle.shp') # For Bark Beetle Mortality


########################################
# Define Global Variables and constants
#########################################

# select for only these years and dates when plotting
plot_years = np.array(range(2018, 2024))
plot_start_date = dt(2018, 1, 1) # Start of time series plot range ### TAG: STD ANOMALY FROM RECONSTRUCTED
plot_end_date = dt(2023, 12, 31) # End of time series plot range

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





def stat_summary(mortality_pixels, control_pixels, dat_var, reference_year = None, shift_years = 0):

   mortality_yr_summary = mortality_pixels.annual_summary_var(dat_var, reference_year, shift_years)
   control_yr_summary = control_pixels.annual_summary_var(dat_var, reference_year, shift_years)

   df = pd.DataFrame({'year': mortality_yr_summary.year.values})
   df['mortality_mean'] = mortality_yr_summary['mean'].values
   df['mortality_std'] = mortality_yr_summary['std'].values
   df['control_mean'] = control_yr_summary['mean'].values
   df['control_std'] = control_yr_summary['std'].values
   df['mean_diff'] = df['mortality_mean'] - df['control_mean']
   df['mannwhitney_p'] = pixel_classes.compare_PixelGroup_MannWhitneyU(mortality_pixels, control_pixels, dat_var, reference_year, shift_years)
   df['kolmogorov_smirnov_p'] = pixel_classes.compare_PixelGroup_Kolmogorov_Smirnov(mortality_pixels, control_pixels, dat_var, reference_year, shift_years)
   df['SMD_simple'] = df['mean_diff'] / np.sqrt((df['mortality_std']**2 + df['control_std']**2) / 2)
   df['n_mortality'] = len(mortality_pixels.pixels)
   df['n_control'] = len(control_pixels.pixels)
   df['pooled_std'] = np.sqrt((((df['n_mortality']-1)*df['mortality_std']**2) + ((df['n_control']-1)*df['control_std']**2)) / (df['n_mortality'] + df['n_control'] - 2))
   df['SMD_pooled'] = df['mean_diff'] / df['pooled_std']
   df['SMD_control'] = df['mean_diff'] / df['control_std']
   
   return df

def compute_kde_density_from_gdf(gdf, xr_dataarray, value_name, value_range, bw_method='scott'):
   """
   Clips the xr_dataarray to the geometry of the gdf, flattens and filters non-NaN values,
   computes the KDE and evaluates the density over value_range.

   Parameters:
      gdf (GeoDataFrame): GeoDataFrame with geometry to clip to.
      xr_dataarray (xr.DataArray or xr.Dataset): xarray data to clip.
      value_name (str): Name of the variable in xr_dataarray to extract.
      value_range (np.ndarray): Range of values to evaluate the KDE.
      bw_method (str or float): Bandwidth method for gaussian_kde.

   Returns:
      kde (scipy.stats.gaussian_kde): Fitted KDE object.
      density (np.ndarray): KDE evaluated over value_range.
      non_nan_values (np.ndarray): Flattened, non-NaN values from the clipped data.
   """
   geom = gdf.geometry.apply(mapping)
   clipped = xr_dataarray.rio.clip(geom, crs=4326)
   values_flat = getattr(clipped, value_name).data.flatten()
   non_nan_mask = ~np.isnan(values_flat)
   non_nan_values = values_flat[non_nan_mask]
   kde = gaussian_kde(non_nan_values, bw_method=bw_method)
   density = kde.evaluate(value_range)
   return kde, density, non_nan_values


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



# Assuming df_array is the array of pandas DataFrames
def calculate_mean_across_dataframes(df_array):
   # Concatenate all DataFrames in the array into a single DataFrame
   concatenated_df = pd.concat(df_array)

   # Group by 'year' and calculate the mean for 'F_statistic' and 'p_value'
   mean_df = concatenated_df.groupby('year').agg({
      'mortality_mean': 'mean',
      'mortality_std': 'mean',
      'control_mean': 'mean',
      'control_std': 'mean',
      'mean_diff': 'median',
      'mannwhitney_p': 'mean',
      'kolmogorov_smirnov_p': 'mean'
   }).reset_index()

   # Assuming mean_df and concatenated_df are already defined
   CI_df = pd.DataFrame({'year': mean_df.year.values})

   # Initialize the columns in CI_df
   CI_df['ci95_lower'] = np.nan
   CI_df['ci95_upper'] = np.nan
   CI_df['ci99_lower'] = np.nan
   CI_df['ci99_upper'] = np.nan
   CI_df['ci99_9_lower'] = np.nan
   CI_df['ci99_9_upper'] = np.nan

   for ii, yyyy in enumerate(CI_df.year.values):
      mean_diffs_yyyy = concatenated_df[concatenated_df['year'] == yyyy]['mean_diff']
      CI_df.loc[ii, 'ci95_lower'], CI_df.loc[ii, 'ci95_upper'] = np.percentile(mean_diffs_yyyy, [2.5, 97.5])
      CI_df.loc[ii, 'ci99_lower'], CI_df.loc[ii, 'ci99_upper'] = np.percentile(mean_diffs_yyyy, [0.5, 99.5])
      CI_df.loc[ii, 'ci99_9_lower'], CI_df.loc[ii, 'ci99_9_upper'] = np.percentile(mean_diffs_yyyy, [0.05, 99.05])

   result_df = pd.merge(mean_df, CI_df, on='year')

   return result_df

# get the y scalevalue in matplotlib over which to set significance asterices above the error bar plot features
def get_yaxis_spacer(ymin, ymax, factor = 0.1):
   return factor * (ymax - ymin)

def add_significance_columns(df):
   def significance_from_pval(row, p_colname):
      if row[p_colname] < 0.001:
         return 3
      elif row[p_colname] < 0.01:
         return 2
      elif row[p_colname] < 0.05:
         return 1
      else:
         return 0
   
   def significance_from_CI(row):
      if (row['ci95_upper'] < 0) | (row['ci95_lower'] > 0):
         # passes with at least alpha = 0.05
         if (row['ci99_upper'] < 0) | (row['ci99_lower'] > 0):
            # passes with at least alpha = 0.01
            if (row['ci99_9_upper'] < 0) | (row['ci99_9_lower'] > 0):
               # passes with alpha = 0.001
               return 3
            else:
               return 2 # didn't pass with alpha = 0.001 but did pass with alpha = 0.01
         else:
            return 1 # didn't pass with alpha = 0.01 but did pass with alpha = 0.05
      else:
         return 0 # didn't pass with alpha = 0.05

   # Apply the function to each row in the DataFrame
   df['mannwhitney_sig'] = df.apply(lambda row: significance_from_pval(row, 'mannwhitney_p'), axis=1)
   df['kolmogorov_smirnov_sig'] = df.apply(lambda row: significance_from_pval(row, 'kolmogorov_smirnov_p'), axis=1)
   df['CI_sig'] = df.apply(significance_from_CI, axis=1)
   
   return df

def plot_summary(summary_df, plot_title, yax_label, mortality_year, fig = None, ax = None, draw_box = False, plot_significance_chars = False):
   summary_df = summary_df[summary_df['year'].isin(plot_years)]

   if fig is None or ax is None:
      # Set up the figure and axes for the map
      fig, ax = plt.subplots(figsize=(3.5, 2)) # establish the figure, axes
   
   ax.grid(color = "gray", linestyle = "--", linewidth = 1, alpha = 0.5, zorder = 1) # add grid lines
   ax.axhline(y = 0, color = 'navy', linestyle = '--', zorder = 2) # dashed line at 0 for context

   # Define the width of the bars and the positions
   width = 0.15
   ind = summary_df.year.values

   mortality_mean = summary_df['mortality_mean'].values
   mortality_std = summary_df['mortality_std'].values
   control_mean = summary_df['control_mean'].values
   control_std = summary_df['control_std'].values

   # Plot the data with error bars
   ax.errorbar(ind - width / 2, mortality_mean, yerr=mortality_std, fmt='o', label='Mortality', capsize=6, color=mortality_color)
   ax.errorbar(ind + width / 2, control_mean, yerr=control_std, fmt='o', label='Control', capsize=6, color=control_color)
   ymin, ymax = ax.get_ylim() # get current y axis limits

   # # Add asterisks for significance
   sigtest_yspacer = get_yaxis_spacer(ymin, ymax)
   # if yax_min is not None and yax_max is not None:
   #    sigtest_yspacer = get_yaxis_spacer(yax_min*0.9, yax_max*0.9)

   # Add gray fill for disturbance period
   if mortality_year < 2024:
      ax.fill_betweenx([-1000, 1000], mortality_year - 0.5, 2024, color='#a4a4a4', alpha=0.4, edgecolor='#6f6f6f', linewidth=1, zorder = 0)


   if draw_box:
      box_left = mortality_year - 0.4
      ax.add_patch(
         plt.Rectangle(
            (box_left, ymin+sigtest_yspacer),  # x, y
            1,
            (ymax - ymin)*0.9,
            facecolor='none',
            edgecolor='white',
            linewidth=2,
            linestyle='--',
            zorder=3
         )
      )

   if plot_significance_chars:
      for i, year in enumerate(summary_df.year.values):
         CI_significance = summary_df.loc[summary_df['year'] == year, 'CI_sig'].values[0]
         MW_significance = summary_df.loc[summary_df['year'] == year, 'mannwhitney_sig'].values[0]
         KS_significance = summary_df.loc[summary_df['year'] == year, 'kolmogorov_smirnov_sig'].values[0]
         max_error = max(mortality_mean[i] + mortality_std[i], control_mean[i] + control_std[i])
            
         if CI_significance > 0:
            asterisks = '*' * CI_significance
            ax.text(year, max_error + sigtest_yspacer, asterisks, ha='center', va='top', color='purple')
            # Check if the new y axis limit exceeds the current ymax
            if max_error + (sigtest_yspacer*2) > ymax:
               ymax = max_error + (sigtest_yspacer*2)

         if MW_significance > 0:
            asterisks = '+' * MW_significance
            ax.text(year, max_error + (sigtest_yspacer*2), asterisks, ha='center', va='top', color='purple')
            # Check if the new y axis limit exceeds the current ymax
            if max_error + (sigtest_yspacer*3) > ymax:
               ymax = max_error + (sigtest_yspacer*3)


   ymax += sigtest_yspacer
   # Add labels for Mortality onset and Mortality detection
   pre_mortality_label_x = mortality_year - 0.65
   mortality_label_x = mortality_year - 0.35
   ax.text(pre_mortality_label_x, ymax-(sigtest_yspacer*0.5), 'Pre-mortality', ha='right', va='top', color='black', fontsize='medium')
   ax.text(mortality_label_x, ymax-(sigtest_yspacer*0.5), 'Mortality', ha='left', va='top', color='black', fontsize='medium')

   # Set the axis labels and title
   ax.set(title=plot_title, ylabel=yax_label)
   ax.set_ylim(ymin, ymax)
   ax.xaxis.label.set_visible(False) # Hide the x-axis label
   ax.set_xticks(summary_df.year.values)
   ax.set_xticklabels(summary_df.year.values)
   ax.set_xlim([plot_start_date.year-0.5, plot_end_date.year+0.5])

   # add legend including the shaded error region
   # fig.legend(loc="upper left", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
   
   # display
   # plt.show()


def plot_bootstrap_CI_stabilization(cumulative_df_arr, yyyy, yax_label, fig = None, ax = None):

   cumulative_df = pd.concat(cumulative_df_arr[1:])

   summary_df = cumulative_df[cumulative_df['year'] == yyyy]

   if fig is None or ax is None:
      fig, ax = plt.subplots(figsize=(6, 5)) # establish the figure, axes
   ax.grid(color = "gray", linestyle = "--", linewidth = 1, alpha = 0.5) # add grid lines
   ax.axhline(y = 0, color = 'navy', linestyle = '--') # dashed line at 0 for context

   ax.plot(summary_df.num_repetitions, summary_df.ci95_upper, color="#bd4c4c")
   ax.plot(summary_df.num_repetitions, summary_df.mean_diff, label='Mean difference', color="#9c0000")
   ax.plot(summary_df.num_repetitions, summary_df.ci95_lower, color="#bd4c4c")
   ax.fill_between(summary_df.num_repetitions, summary_df.ci95_lower, summary_df.ci95_upper, color="#bd4c4c", alpha=0.7, label='95% CI')

   # Define the axis, title labels
   ax.set(title=rf'1yr $\Delta$SIF, year {yyyy}' + '\nBootstrap 95% confidence interval', ylabel=yax_label, xlabel='num repetitions')

   if np.nanmax(summary_df.num_repetitions) < 10:
      ax.set_xlim([2, 10])

   # add legend including the shaded error region
   ax.legend()


def plot_bootstrap_pval_stabilization(cumulative_df_arr, yyyy,  fig = None, ax = None):
   cumulative_df = pd.concat(cumulative_df_arr[1:])
   # unique_years = cumulative_df.year.unique()
   summary_df = cumulative_df[cumulative_df['year'] == yyyy]

   if fig is None or ax is None:
      fig, ax = plt.subplots(figsize=(6, 5)) # establish the figure, axes
   ax.grid(color = "gray", linestyle = "--", linewidth = 1, alpha = 0.5) # add grid lines

   ax.plot(summary_df.num_repetitions, summary_df.mannwhitney_p, label='running mean', color='#e41a1c')
   # ax.plot(summary_df.num_repetitions, summary_df.kolmogorov_smirnov_p, label='Kolmogorov-Smirnov', color='#377eb8')

   plt.axhline(0.05, linestyle = '--', color = '#a4a4a4', label='p = 0.05')
   plt.axhline(0.01, linestyle = '--', color = '#525252', label='p = 0.01')

   # Define the axis, title labels
   ax.set(title=rf'1yr $\Delta$SIF, year {yyyy}' + '\nMann-Whitney Rank Sum cumulative mean p-value', ylabel='p', xlabel='num repetitions')

   if np.nanmax(summary_df.num_repetitions) < 10:
      ax.set_xlim([2, 10])

   # add legend including the shaded error region
   ax.legend()

