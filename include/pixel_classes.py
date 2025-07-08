#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Lewis Kunik - University of Utah
# Contact: lewis.kunik@utah.edu
#

#########################################
# Load packages
#########################################

# numerical/data management packages
import numpy as np
import xarray as xr
from scipy import stats



#########################################
# Define Global Variables
#########################################


#########################################
# Define Global Functions
#########################################

# Check whether the input xarray object contains the pixel within its x, y bounds
# returns True if pixel coordinates are found inside the xarray object, otherwise False
# Note xarray object coords must be labeled as either x,y or lon,lat
def check_pixel_inside_xr_bounds(pixel, xr_da):
      # Extract coordinates from the DataArray
      x_coords = xr_da.lon.values if 'lon' in xr_da.coords else xr_da.x.values
      y_coords = xr_da.lat.values if 'lat' in xr_da.coords else xr_da.y.values

      # Estimate the dataset xy bounds, assuming spatial resolution is consistent across all lat and across all lon
      x_bnd_min = np.nanmin(x_coords) - (abs(x_coords[1] - x_coords[0]) / 2)
      x_bnd_max = np.nanmax(x_coords) + (abs(x_coords[1] - x_coords[0]) / 2)
      y_bnd_min = np.nanmin(y_coords) - (abs(y_coords[1] - y_coords[0]) / 2)
      y_bnd_max = np.nanmax(y_coords) + (abs(y_coords[1] - y_coords[0]) / 2)

      # print('xr.DataArray:')
      # print(f'x bounds = ({round(x_bnd_min, 5)}, {round(x_bnd_max, 5)})')
      # print(f'y bounds = ({round(y_bnd_min, 5)}, {round(y_bnd_max, 5)})')

      # Check if the dataset bounds include this pixel (if not, raise exception)
      if (pixel.center_lon < x_bnd_min) | \
         (pixel.center_lon > x_bnd_max) | \
         (pixel.center_lat < y_bnd_min) | \
         (pixel.center_lat > y_bnd_max):
         print(f'OUT OF BOUNDS: xr_da x bounds = ({x_bnd_min}, {x_bnd_max}), y bounds = ({y_bnd_min}, {y_bnd_max}); Pixel centroid = ({pixel.center_lon}, {pixel.center_lat})')
         return False
      else:
         return True



#########################################
# Define classes
#########################################

########################################################
###~~~~~~~~~~~~~~~~~~~~~~ Pixel ~~~~~~~~~~~~~~~~~~~~~###
########################################################

class Pixel:
   def __init__(self, center_lon: float, center_lat: float, deg_res: float):
      self.center_lon = center_lon
      self.center_lat = center_lat
      self.deg_res = deg_res
      margin = deg_res / 50

      # keep track of the pixel grid box bounds
      self.min_lon = center_lon - (deg_res/2) - margin
      self.max_lon = center_lon + (deg_res/2) + margin
      self.min_lat = center_lat - (deg_res/2) - margin
      self.max_lat = center_lat + (deg_res/2) + margin

      # Dictionary to hold multiple xarray.DataArray instances
      self.data_vars = {}
      self.data_vars_annual_summary = {}

   def __repr__(self):
      data_var_names = list(self.data_vars.keys())
      annual_summary_names = list(self.data_vars_annual_summary.keys())
      return (f"Pixel centroid: lon = {round(self.center_lon, 5)}, lat = {round(self.center_lat, 5)}, "
               f"resolution = {round(self.deg_res, 4)}°, "
               f"data_vars={data_var_names}), "
               f"annual summaries for {annual_summary_names}")

   def get_data_var(self, var_name: str) -> xr.DataArray:
      return self.data_vars.get(var_name, None)

   def get_data_var_annual_summary(self, var_name: str) -> xr.DataArray:
      return self.data_vars_annual_summary.get(var_name, None)

   # Concatenate additional data to the timeseries of data contained in the DataVar
   def concat_data(self, var_name, additional_data_array: xr.DataArray):
      if var_name not in list(self.data_vars.keys()):
         raise Exception(f'variable {var_name} not found in existing dictionary')

      if 'time' not in additional_data_array.dims:
         raise Exception('Additional DataArray must have a time dimension')
      
      self.data_vars[var_name] = xr.concat([self.data_vars[var_name], additional_data_array], dim='time').sortby('time')
      # print(f'dtype is {type(self.data_vars[var_name])}')

   def add_data_from_xr(self, xr_da: xr.DataArray, var_name: str, units: str = None):
      """
      Adds data from an xarray DataArray to the Pixel's data_vars attribute.
      The DataArray is expected to have dimensions (x, y, time) or (lon, lat, time).
      
      Args:
         xr_da (xarray.DataArray): Input DataArray containing the data.
         var_name (str): Name of the variable to be stored in data_vars.
         units (str, optional): Units of the data variable.
      """

      # Ensure the selected data only has the time dimension
      if 'time' not in xr_da.dims:
         raise Exception('Selected data do not have a time dimension. Exiting..')

      if not check_pixel_inside_xr_bounds(self, xr_da):
         raise Exception('Pixel coordinate is out of dataarray bounds. Exiting..')

      # Select the data for the pixel's bounds (average over all grid cells within the pixel bounds)
      if ('lon' in xr_da.dims) & ('lat' in xr_da.dims):
         pixel_data = xr_da.sel(
         lon=slice(self.min_lon, self.max_lon),
         lat=slice(self.min_lat, self.max_lat)
         ).mean(dim=[d for d in ['lon', 'lat'] if d in xr_da.dims])
      else:
         pixel_data = xr_da.sel(
         x=slice(self.min_lon, self.max_lon),
         y=slice(self.min_lat, self.max_lat)
         ).mean(dim=[d for d in ['x', 'y'] if d in xr_da.dims])
         
      # if variable already exists in the dictionary, concatenate with existing
      
      if var_name in list(self.data_vars.keys()):
         self.concat_data(var_name, pixel_data) # compute needed to set to memory from dask array
         # print(f'concatenating - dtype is {type(pixel_data)}')
      else:
         # Otherwise add directly to dictionary
         self.data_vars[var_name] = pixel_data # compute needed to set to memory from dask array
         # print(f'first instance - dtype is {type(pixel_data)}')

   def filter_var_by_months(self, varname_to_filter: str, months_to_remove):
      
      # Check that both variables exist in the dictionary
      if varname_to_filter not in list(self.data_vars.keys()):
         raise Exception(f'variable {varname_to_filter} not found in existing dictionary')
      if any([(mm not in np.arange(1, 13)) for mm in months_to_remove]):
         raise ValueError('provided months are not in the range of 1-12')
      
      discardmonthsTF = np.isin(self.data_vars[varname_to_filter]['time'].dt.month, months_to_remove)
      self.data_vars[varname_to_filter] = self.data_vars[varname_to_filter].where(~discardmonthsTF)


   def summarize_annual(self, var_name: str, method = 'average'):

      # Check that variable exists in the dictionary
      if var_name not in list(self.data_vars.keys()):
         raise Exception(f'variable {var_name} not found in existing dictionary')
      
      temp_var = self.data_vars[var_name]
      if method == 'average':
         self.data_vars_annual_summary[var_name] = temp_var.groupby('time.year').mean(skipna=True)
      if method == 'sum':
         self.data_vars_annual_summary[var_name] = temp_var.groupby('time.year').sum(skipna=True)



########################################################
###~~~~~~~~~~~~~~~~~~~ PixelGroup ~~~~~~~~~~~~~~~~~~~###
########################################################

class PixelGroup:
   def __init__(self):
      self.pixels = []

   def __repr__(self):
      repr_string_arr = [f'Pixel Group containing {len(self.pixels)} pixels']
      if len(self.pixels) < 15:
         for pixel in self.pixels:
            repr_string_arr.append(str(pixel))
      elif len(self.pixels) < 50:
         for ii, pixel in enumerate(self.pixels):
            repr_string_arr.append(f'pixel {ii}: lon = {pixel.center_lon}, lat = {pixel.center_lat}')
      return "\n".join(repr_string_arr)

   def print_all_pixels_info(self):
      for pixel in self.pixels:
         for key, value in pixel.data_vars.items():
            print(f'{value}')

   def print_all_pixels_annual_summary(self):
      for pixel in self.pixels:
         for key, value in pixel.data_vars_annual_summary.items():
            print(f'{value}')

   def add_pixel(self, pixel: Pixel):
      self.pixels.append(pixel)

   def get_unique_data_vars(self):
      unique_data_vars = {}
      for pixel in self.pixels:
         for var_name, data_array in pixel.data_vars.items():
               if var_name not in unique_data_vars:
                  unique_data_vars[var_name] = data_array.attrs.get('units', None)
      return unique_data_vars

   def get_unique_annual_summary_vars(self):
      unique_data_vars = {}
      for pixel in self.pixels:
         for var_name, data_array in pixel.data_vars_annual_summary.items():
               if var_name not in unique_data_vars:
                  unique_data_vars[var_name] = data_array.attrs.get('units', None)
      return unique_data_vars

   # returns array with [min_lon, max_lon, min_lat, max_lat]
   def get_total_lonlat_bounds(self):

      min_lon = 180
      max_lon = -180
      min_lat = 90
      max_lat = -90

      for pixel in self.pixels:
         min_lon = np.min(min_lon, pixel.min_lon)
         max_lon = np.max(max_lon, pixel.max_lon)
         min_lat = np.min(min_lat, pixel.min_lat)
         max_lat = np.max(max_lat, pixel.max_lat)
      
      return [min_lon, max_lon, min_lat, max_lat]

   def add_xr_to_all_pixels(self, xr_da: xr.DataArray, var_name: str, units: str = None):
      for pixel in self.pixels:
         pixel.add_data_from_xr(xr_da, var_name, units)
   
   def filter_var_by_months(self, varname_to_filter: str, months_to_remove):
      for pixel in self.pixels:
         pixel.filter_var_by_months(varname_to_filter, months_to_remove)


   def summarize_annual(self, var_name: str, method = 'average'):
      for pixel in self.pixels:
         pixel.summarize_annual(var_name, method)

   # Assumes all pixels' data have the same time dimension (because come from same dataset)
   def summary_var(self, var_name):

      all_data = []
      for ii, pixel in enumerate(self.pixels):
         if var_name in pixel.data_vars:
            xr_da_to_append = pixel.data_vars[var_name].expand_dims(pixel = [ii])
            all_data.append(xr_da_to_append)

      pixels_combined = xr.concat(all_data, dim = 'pixel')
      summary_mean = pixels_combined.mean(dim = 'pixel', skipna = True)
      summary_std = pixels_combined.std(dim = 'pixel', skipna = True)
      combined_data = xr.Dataset({
         'mean': summary_mean,
         'std': summary_std
      })

      return combined_data
   
   # Assumes all pixels' data have the same time dimension (because come from same dataset)
   def annual_summary_var(self, var_name, reference_year = None, shift_years = 0):

      all_data = []
      for ii, pixel in enumerate(self.pixels):
         if var_name in pixel.data_vars_annual_summary:
            xr_da = pixel.data_vars_annual_summary[var_name]
            if reference_year is not None:
               da_diff = xr_da - xr_da.sel(year=reference_year) # difference from reference year
               xr_da_to_append = da_diff.expand_dims(pixel = [ii])
            else:
               if shift_years == 0:
                  xr_da_to_append = xr_da.expand_dims(pixel = [ii])
               else:
                  da_diff = xr_da - xr_da.shift(year=shift_years) # difference from X years prior
                  xr_da_to_append = da_diff.expand_dims(pixel = [ii])
            all_data.append(xr_da_to_append)

      pixels_combined = xr.concat(all_data, dim = 'pixel')
      summary_mean = pixels_combined.mean(dim = 'pixel', skipna = True)
      summary_std = pixels_combined.std(dim = 'pixel', skipna = True)
      combined_data = xr.Dataset({
         'mean': summary_mean,
         'std': summary_std
      })

      return combined_data
   


########################################################
###~~~~~~~~~ Functions with PixelGroups ~~~~~~~~~###
########################################################


def compare_PixelGroup_MannWhitneyU(pixel_group1, pixel_group2, var_name, reference_year = None, shift_years = 0):

   # Check that the variable exists in the data_vars of both pixel groups
   if var_name not in pixel_group1.get_unique_annual_summary_vars():
      raise Exception(f'annual variable {var_name} not found in PixelGroup1')
   if var_name not in pixel_group2.get_unique_annual_summary_vars():
      raise Exception(f'annual variable {var_name} not found in PixelGroup2')

   # Get the data arrays for the variable
   pixel_group_data1 = []
   pixel_group_data2 = []
   for ii, pixel in enumerate(pixel_group1.pixels):
      xr_da = pixel.data_vars_annual_summary[var_name]
      if reference_year is not None:
         da_diff = xr_da - xr_da.sel(year=reference_year) # difference from reference year
         xr_da_to_append = da_diff
      else:
         if shift_years == 0:
            xr_da_to_append = xr_da
         else:
            da_diff = xr_da - xr_da.shift(year=shift_years) # difference from X years prior
            xr_da_to_append = da_diff
      pixel_group_data1.append(xr_da_to_append.expand_dims(pixel_num = [ii]))
   for ii, pixel in enumerate(pixel_group2.pixels):
      xr_da = pixel.data_vars_annual_summary[var_name]
      if reference_year is not None:
         da_diff = xr_da - xr_da.sel(year=reference_year) # difference from reference year
         xr_da_to_append = da_diff
      else:
         if shift_years == 0:
            xr_da_to_append = xr_da
         else:
            da_diff = xr_da - xr_da.shift(year=shift_years) # difference from X years prior
            xr_da_to_append = da_diff
      pixel_group_data2.append(xr_da_to_append.expand_dims(pixel_num = [ii]))

   # Concatenate the data arrays
   pixels_combined1 = xr.concat(pixel_group_data1, dim = 'pixel_num')
   pixels_combined2 = xr.concat(pixel_group_data2, dim = 'pixel_num')
   
   # Perform the Mann-Whitney U test by year
   p_values_by_year = []
   for yyyy in pixels_combined1.year.values:
      data1 = pixels_combined1.sel(year=yyyy).values
      data2 = pixels_combined2.sel(year=yyyy).values
      stat, p = stats.mannwhitneyu(data1, data2)
      p_values_by_year.append(p)
  
   return p_values_by_year

def compare_PixelGroup_Kolmogorov_Smirnov(pixel_group1, pixel_group2, var_name, reference_year = None, shift_years = 0):

   # Check that the variable exists in the data_vars of both pixel groups
   if var_name not in pixel_group1.get_unique_annual_summary_vars():
      raise Exception(f'annual variable {var_name} not found in PixelGroup1')
   if var_name not in pixel_group2.get_unique_annual_summary_vars():
      raise Exception(f'annual variable {var_name} not found in PixelGroup2')

   # Get the data arrays for the variable
   pixel_group_data1 = []
   pixel_group_data2 = []
   for ii, pixel in enumerate(pixel_group1.pixels):
      xr_da = pixel.data_vars_annual_summary[var_name]
      if reference_year is not None:
         da_diff = xr_da - xr_da.sel(year=reference_year) # difference from reference year
         xr_da_to_append = da_diff
      else:
         if shift_years == 0:
            xr_da_to_append = xr_da
         else:
            da_diff = xr_da - xr_da.shift(year=shift_years) # difference from X years prior
            xr_da_to_append = da_diff
      pixel_group_data1.append(xr_da_to_append.expand_dims(pixel_num = [ii]))
   for ii, pixel in enumerate(pixel_group2.pixels):
      xr_da = pixel.data_vars_annual_summary[var_name]
      if reference_year is not None:
         da_diff = xr_da - xr_da.sel(year=reference_year) # difference from reference year
         xr_da_to_append = da_diff
      else:
         if shift_years == 0:
            xr_da_to_append = xr_da
         else:
            da_diff = xr_da - xr_da.shift(year=shift_years) # difference from X years prior
            xr_da_to_append = da_diff
      pixel_group_data2.append(xr_da_to_append.expand_dims(pixel_num = [ii]))

   # Concatenate the data arrays
   pixels_combined1 = xr.concat(pixel_group_data1, dim = 'pixel_num')
   pixels_combined2 = xr.concat(pixel_group_data2, dim = 'pixel_num')
   
   # Perform the Kolmogorov-Smirnov test by year
   p_values_by_year = []
   for yyyy in pixels_combined1.year.values:
      data1 = pixels_combined1.sel(year=yyyy).values
      data2 = pixels_combined2.sel(year=yyyy).values
      stat, p = stats.ks_2samp(data1, data2)
      p_values_by_year.append(p)
  
   return p_values_by_year