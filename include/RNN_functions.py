#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Lewis Kunik - University of Utah
# Contact: lewis.kunik@utah.edu



#########################################
# Load packages
#########################################

#%%


# runtime packages
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# numerical/data management packages
import numpy as np

# time and date packages
from datetime import datetime as dt  # date time library

# machine learning packages
import tensorflow as tf

#########################################
# Define Global Filepaths
#########################################


########################################
# Define Global Variables and constants
#########################################


########################################
# Define Functions
#########################################


def check_all_lengths_in_arr(arr):

   # Check if all elements in arr have the same length
   arr_lengths = [len(ts) for ts in arr] 
   if len(set(arr_lengths)) == 1:
      print(f"All elements in input array have the same length {set(arr_lengths)}.")
   else:
      print("Elements in input array have different lengths.")
      print(f"Lengths: {set(arr_lengths)}")

# Custom attention layer for RNN
class AttentionLayer(tf.keras.layers.Layer):
   def __init__(self, **kwargs):
      super(AttentionLayer, self).__init__(**kwargs)
      
   def build(self, input_shape):
      self.W = self.add_weight(name="attention_weight", shape=(input_shape[-1], 1),
                              initializer="glorot_uniform", trainable=True)
      self.b = self.add_weight(name="attention_bias", shape=(input_shape[1], 1),
                              initializer="zeros", trainable=True)
      super(AttentionLayer, self).build(input_shape)
   
   def call(self, x):
      # Calculate attention scores
      e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
      
      # Calculate attention weights
      a = tf.nn.softmax(e, axis=1)
      
      # Apply attention weights to input
      context = x * a
      
      return context
   
   def compute_output_shape(self, input_shape):
      return input_shape

# Function to create a seasonal attention mask
def create_attention_mask(sequence_length, recent_months=3):
   """
   Creates an attention mask that increases linearly for the most recent months
   
   Parameters:
   - sequence_length: Length of the time series sequence
   - recent_months: Number of recent months to emphasize
   
   Returns:
   - attention_mask: Attention mask tensor of shape [1, sequence_length, 1]
   """
   # Initialize with ones
   attention_mask = np.ones((1, sequence_length, 1))
   
   # Calculate number of time steps for n months
   recent_points = recent_months*4
   
   # Increase attention weights for recent data
   if recent_points > 0:
      # Linear increase from 1 to 3 for the most recent points
      end_weights = np.linspace(1.0, 3.0, recent_points).reshape(-1, 1)
      attention_mask[0, -recent_points:, 0] = end_weights.flatten()
   

   return tf.constant(attention_mask, dtype=tf.float32)

# stacked_xr = mortality_xr_stacked
# num_samples = num_sample_pixels
# random_seed = 42
# Stacked_xr must have NIRv, elevation and tree_canopy_cover variables
def sample_xarray_stacked(stacked_xr, indices_to_sample, ts_start_date=dt(2001, 1, 1), ts_end_date=dt(2021, 10, 1)):
   """
   Randomly sample time series from a stacked xarray dataset
   
   Parameters:
   - xr_obj: An xarray DataArray with time dimension
   - num_samples: Number of time series to sample
   - random_seed: Random seed for reproducibility
   
   Returns:
   - time_series_array: numpy array of time series
   - timestamps: corresponding timestamps
   - coords: coordinates of sampled pixels
   """
   

   # Extract time series and coordinates
   dat_values_list = []
   elev_list = []
   TCC_list = []
   timestamps_list = []
   coords_list = []
   
   for idx in indices_to_sample:
      pixel_data = stacked_xr.isel(pixel=idx)

      # Filter pixel data for values after ts_start_date and before ts_end_date
      time_mask = (pixel_data['time'].values >= np.datetime64(ts_start_date)) & (pixel_data['time'].values <= np.datetime64(ts_end_date))
      pixel_data = pixel_data.sel(time=time_mask)
      timestamps_list.append(pixel_data['time'].values)
      dat_values_list.append(pixel_data['NIRv'].values)
      elev_list.append(pixel_data['elevation'].values)
      TCC_list.append(pixel_data['tree_canopy_cover'].values)

      # Get coordinates
      y_coord, x_coord = pixel_data['y'].values, pixel_data['x'].values
      coords_list.append((y_coord, x_coord))
   
   # Convert to numpy arrays
   dat_values_array = np.array(dat_values_list, dtype="object")
   elev_array = np.array(elev_list)
   TCC_array = np.array(TCC_list)
   timestamps_array = np.array(timestamps_list)
   coords = np.array(coords_list)
   
   return dat_values_array, elev_array, TCC_array, timestamps_array, coords


def cap_and_replace_inf_nan(arr, min_val=-25, max_val=25):
   """
   Cap values in the input array/list:
   - Replace -inf with min_val, +inf with max_val
   - Replace NaN with 0
   - Cap all values to [min_val, max_val]
   Works for list of arrays or 2D numpy arrays.
   """
   capped_arr = []
   for ts in arr:
      ts = np.array(ts, dtype=float)
      ts[np.isneginf(ts)] = min_val
      ts[np.isposinf(ts)] = max_val
      ts = np.clip(ts, min_val, max_val)
      ts[np.isnan(ts)] = 0
      capped_arr.append(ts)
   return np.array(capped_arr, dtype=object)

