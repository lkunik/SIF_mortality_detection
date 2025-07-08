#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Lewis Kunik - University of Utah
# Contact: lewis.kunik@utah.edu
#

#########################################
# Load packages
#########################################

# # File system packages
import os  # operating system library

# # numerical/data management packages
import numpy as np

from datetime import datetime as dt

#########################################
# Define Global Filepaths
#########################################

elev_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/SRTM/WUS/'
elev_file = os.path.join(elev_dir, 'SRTM_DEM_05d_WUS.nc')

TCC_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/NLCD/TreeCover/2021/'
TCC_file = os.path.join(TCC_dir, 'NLCD_TCC_05d_WUS_2021.nc')

LAI_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/MODIS/LAI_MOD15A2H/'
LAI_July_file = os.path.join(LAI_dir, 'MODIS_LAI_WUS_05d_2011-2023_July_mean.nc')


#########################################
# CASE DETAILS
#########################################
# Number of bootstrapping iterations to loop through
num_repeats = 500

mortality_detection_year = 2020
include_months_lab = 'Aug-Oct'
include_months_descr = 'late_summer'
discard_months = [11, 12, 1, 2, 3, 4, 5, 6, 7] # define "winter" months to discard (months when snowcover may affect signal)

# number of control pixels to sample from the KDE distribution for each bootstrap iteration
# NOTE: must run 10d_test_control_domain_elevation_KDE.py from disturbance code dir to determine this value
control_sample_size = 100
override_mortality_sample_size = False # Usually keep this as False
mortality_sample_size = 13

########## Filepaths to mortality/control shapefiles:

# ## BARK BEETLE
# disturbance_type = 'Detection'
# severity_level = 'ModerateSevere'
# est_mortality_year = mortality_detection_year - 1
# simulation_ID = f'boot_{include_months_descr}_{include_months_lab}_{severity_level}{mortality_detection_year}' # For Bark Beetle Mortality
# summary_plot_titlebase = rf'{mortality_detection_year} ADS Mortality vs Control; Elevation\+TCC KDE filter;{include_months_descr} ({include_months_lab}) Analysis ({num_repeats} reps)'
# control_polygon_basedir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/disturbance/output/shp/05d/base/' # For Bark Beetle Mortality
# control_polygon_filepath = os.path.join(control_polygon_basedir,'control_pixels_base_2011-2023_cumMA_lt10p_ENF_noBurn_SCKlower_05d.shp') # For Bark Beetle Mortality
# mortality_polygon_basedir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/disturbance/output/shp/05d/ModerateSevere/by_year' # For Bark Beetle Mortality
# mortality_polygon_filepath =  os.path.join(mortality_polygon_basedir, 'mortality_pixels_2023_ModerateSevere_MA_25-35p_05d.shp') # For Bark Beetle Mortality


### WILDFIRE
disturbance_type = 'Burn' # MUST BE 'Burn'
est_mortality_year = mortality_detection_year
simulation_ID = f'boot_{include_months_descr}_{include_months_lab}_Wildfires{mortality_detection_year}' # For Wildfire Mortality
control_polygon_basedir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/disturbance/output/shp/05d/wildfire/base/' # For Wildfire Mortality
control_polygon_filepath = os.path.join(control_polygon_basedir,'control_pixels_base_2011-2023_cumMA_lt10p_ENF_noBurn_SCK_05d.shp') # For Wildfire Mortality
mortality_polygon_basedir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/disturbance/output/shp/05d/wildfire/base/by_year' # For Wildfire Mortality
summary_plot_titlebase = rf'{mortality_detection_year} Wildfires vs Control; Elevation\+TCC KDE filter;{include_months_descr} ({include_months_lab}) Analysis ({num_repeats} reps)'
mortality_polygon_filepath =  os.path.join(mortality_polygon_basedir, f'wildfire_pixels_base_{mortality_detection_year}_pixelBurn_gt0.95_ENF_SCK_05d.shp') # For Wildfire Mortality


########################################
# Define Other Global Variables and constants
#########################################

### Define winter months, a threshold for snowcover mask, threshold for filtering out low-productivity summer values
snow_thresh = 0.25
filter_quantile = 0 # above the Xth percentile within the "non-winter" months

ensemble_local_outdir = simulation_ID # name of directory where output will be saved
ensemble_local_pixelmap_dir = f'{simulation_ID}_pixel_maps' # name of directory where pixel maps will be saved
ensemble_local_plotdir = simulation_ID # name of directory where output plots will be saved
ensemble_local_validation_plotdir = f'{simulation_ID}_stats'
summary_filename_base = simulation_ID

plot_pixel_maps = False # T/F, should you plot maps showing pixel locations for each bootstrap iteration?
plot_bootstrap_validation = False # T/F, should you plot values vs. number of realizations to ensure stabilization of mean?
pixel_map_tiler_zoom = 7  # define a zoom level of detail
pixel_map_extent =  [-125, -117, 35, 45]  # [minx, maxx, miny, maxy], bounds for western US

# Define the mortality detection years to analyze (e.g. 2022-2023)
mortality_years = np.array(range(2022, 2024))

# Define parameters for Kernel Density Estimation of land surface characteristic distributions
elevation_bin = 100 # meters
elevation_range = np.linspace(0, 4000, 1000)#[:, None]  #uncomment the [:, None] if using 1D KDE
LAI_bin = 0.25 # unitless
LAI_range = np.linspace(0, 8, 1000)#[:, None] #uncomment the [:, None] if using 1D KDE
TCC_bin = 5 # percent
TCC_range = np.linspace(0, 100, 1000)#[:, None] #uncomment the [:, None] if using 1D KDE

# select for only these years when working with data files
analysis_years = np.array(range(2015, 2024))

# SPEI_month_timescales = [12, 24, 36, 48] # 1, 2, 3, 6, 9, 12, 24, 36, 48, 60, 72]

### Plotting options:
mortality_color = "#aa021e" # Line/point color for plots
control_color = "#0219aa"

# Summary plot labels and info
Boot_CI_sig_char = '*' # character to denote significance in summary plots
MannWhitney_sig_char = '+' # character to denote significance in summary plots
KolSmir_sig_char = '' # character to denote significance in summary plots

# Effect size max and mins
effect_size_xmax = 3.54
effect_size_xmin = -3.54
PRB_xmin = -175
PRB_xmax = 175

override_summary_plot_yax = False
override_yax_absmean = [0.002297,0.4313596]
override_yax_1yrdiff = [-0.17124,0.165137]
#########################################
# Define Global Functions
#########################################

def save_config_summary_file(outdir):
   config_details = {
      "simulation ID": simulation_ID,
      "description": summary_plot_titlebase,
      "control_polygon_filepath": control_polygon_filepath,
      "mortality_polygon_filepath": mortality_polygon_filepath,
      "bootstrap control resample size": control_sample_size,
      "discard months": discard_months,
      "snow threshold": snow_thresh,
      "percentile cutoff for summer values": filter_quantile,
      "bootstrap repititions": num_repeats
   }
   
   with open(os.path.join(outdir, 'config_details.txt'), 'w') as f:
      for key, value in config_details.items():
         f.write(f"{key}: {value}\n")