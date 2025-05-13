import numpy as np
import pandas as pd
import os
import logging
import time
import datetime
from datetime import date
from google.cloud import bigquery

from meridian.data import input_data
from meridian.data import load
from meridian import constants
import xarray as xr

# --- Helper function for Time Conversion ---
def convert_time_column_to_string(df_to_convert: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Converts the specified column in the DataFrame to 'YYYY-MM-DD' string format."""
    if col_name in df_to_convert.columns:
        logging.info(f"Converting time column '{col_name}' to string format 'YYYY-MM-DD'")
        # Check if the column is already datetime-like or a date object
        if pd.api.types.is_datetime64_any_dtype(df_to_convert[col_name]) or \
            (not df_to_convert[col_name].empty and isinstance(df_to_convert[col_name].iloc[0], (pd.Timestamp, datetime.date))):
            df_to_convert[col_name] = pd.to_datetime(df_to_convert[col_name]).dt.strftime('%Y-%m-%d')
            logging.info(f"Conversion of '{col_name}' complete.")
        elif pd.api.types.is_string_dtype(df_to_convert[col_name]):
            logging.info(f"Column '{col_name}' is already string type. Checking format (first row): {df_to_convert[col_name].iloc[0] if not df_to_convert[col_name].empty else 'empty column'}")
            # Optional: Add validation here to ensure it's already 'YYYY-MM-DD'
        else:
            logging.warning(f"Column '{col_name}' is not a recognized datetime or string type ({df_to_convert[col_name].dtype}). Meridian might still fail.")
    else:
        logging.error(f"Specified time column '{col_name}' not found in DataFrame!")
        raise ValueError(f"Time column '{col_name}' not found in BigQuery results.")
    return df_to_convert
# --- End Helper function ---


# --- Helper function for Time Outlier Detection ---
def detect_and_log_time_outliers(df_to_check: pd.DataFrame, col_name: str, min_date_str: str, max_date_str: str):
    """Detects and logs outliers in a time column based on an expected date range."""
    logging.info(f"Performing outlier detection on time column '{col_name}'...")
    try:
        min_expected_date = pd.to_datetime(min_date_str)
        max_expected_date = pd.to_datetime(max_date_str)

        # Convert the string time column to datetime objects for comparison
        # Assumes the column is already in 'YYYY-MM-DD' string format
        df_time_as_datetime = pd.to_datetime(df_to_check[col_name], format='%Y-%m-%d', errors='coerce')

        # Identify outliers
        outliers = df_to_check[(df_time_as_datetime < min_expected_date) | (df_time_as_datetime > max_expected_date) | (df_time_as_datetime.isna())]

        if not outliers.empty:
            logging.warning(f"Found {len(outliers)} rows with time values outside the expected range [{min_date_str} - {max_date_str}] or unparseable dates.")
            logging.warning(f"First 5 outliers:\n{outliers[[col_name]].head().to_string()}")
        else:
            logging.info(f"No time outliers detected outside the range [{min_date_str} - {max_date_str}].")
    except Exception as e:
        logging.error(f"Error during time outlier detection: {e}")
    return
# --- End Helper function ---


# --- Helper function for Non-Negative Value Check ---
def check_and_log_zero_negative_values(df_to_check: pd.DataFrame, columns_to_validate: list[str]):
    """Checks specified numeric columns for zero or negative values and logs warnings."""
    logging.info(f"Performing zero or negative value checks for columns: {columns_to_validate}")
    for col_name in columns_to_validate:
        if col_name not in df_to_check.columns:
            logging.warning(f"Column '{col_name}' specified for non-negative check not found in DataFrame. Skipping.")
            continue

        # Ensure column is numeric; coercing errors will turn non-numeric to NaN.
        # NaNs will not satisfy the condition <= 0.
        try:
            numeric_col = pd.to_numeric(df_to_check[col_name], errors='coerce')
            
            # Identify rows where the numeric value is <= 0
            # NaNs in numeric_col (due to coercion or original NaNs) will result in False for the comparison.
            problematic_rows_df = df_to_check[numeric_col <= 0]

            if not problematic_rows_df.empty:
                logging.warning(
                    f"Column '{col_name}' has {len(problematic_rows_df)} rows with zero or negative values. "
                    f"Example problematic values: {problematic_rows_df[col_name].head().tolist()}"
                )
            else:
                logging.info(f"Column '{col_name}' has no zero or negative values.")
        except Exception as e:
            logging.error(f"Error during non-negative check for column '{col_name}': {e}. Original dtype: {df_to_check[col_name].dtype}")
    logging.info("Zero or negative value checks complete.")
# --- End Helper function ---


# --- Helper function for columns in coord_to_columns mapping Check ---
def check_coord_to_columns_mapping(df_to_check: pd.DataFrame, coord_to_columns: dict[str, list[str]]):
    """Checks if columns defined in the mapping are in the dataframe"""
    for c in coord_to_columns.media:
      if c not in df_to_check.columns:
        logging.error(f"Column '{c}' not found in the CSV file.")
        
    for c in coord_to_columns.controls + [
        coord_to_columns.time,
        coord_to_columns.geo,
        coord_to_columns.kpi,
        coord_to_columns.revenue_per_kpi,
    ]:
      if c not in df_to_check.columns:
        print(f"Column '{c}' not found in the CSV file.")


def reload_data(self) -> input_data.InputData:
    """Reads data from a dataframe and returns an InputData object."""

    # Change geo strings to numbers to keep the order of geos. The .to_xarray()
    # method from Pandas sorts lexicographically by the key columns, so if the
    # geos were unsorted strings, it would change their order.
    geo_column_name = self.coord_to_columns.geo
    time_column_name = self.coord_to_columns.time
    geo_names = self.df[geo_column_name].unique()
    self.df[geo_column_name] = self.df[geo_column_name].replace(
        dict(zip(geo_names, np.arange(len(geo_names))))
    )
    df_indexed = self.df.set_index([geo_column_name, time_column_name])

    kpi_xr = (
        df_indexed[self.coord_to_columns.kpi]
        .dropna()
        .rename(constants.KPI)
        .rename_axis([constants.GEO, constants.TIME])
        .to_frame()
        .to_xarray()
    )
    population_xr = (
        df_indexed[self.coord_to_columns.population]
        .groupby(geo_column_name)
        .mean()
        .rename(constants.POPULATION)
        .rename_axis([constants.GEO])
        .to_frame()
        .to_xarray()
    )
    controls_xr = (
        df_indexed[self.coord_to_columns.controls]
        .stack()
        .rename(constants.CONTROLS)
        .rename_axis(
            [constants.GEO, constants.TIME, constants.CONTROL_VARIABLE]
        )
        .to_frame()
        .to_xarray()
    )
    dataset = xr.combine_by_coords([kpi_xr, population_xr, controls_xr])

    if self.coord_to_columns.non_media_treatments is not None:
      non_media_xr = (
          df_indexed[self.coord_to_columns.non_media_treatments]
          .stack()
          .rename(constants.NON_MEDIA_TREATMENTS)
          .rename_axis(
              [constants.GEO, constants.TIME, constants.NON_MEDIA_CHANNEL]
          )
          .to_frame()
          .to_xarray()
      )
      dataset = xr.combine_by_coords([dataset, non_media_xr])

    if self.coord_to_columns.revenue_per_kpi is not None:
      revenue_per_kpi_xr = (
          df_indexed[self.coord_to_columns.revenue_per_kpi]
          .dropna()
          .rename(constants.REVENUE_PER_KPI)
          .rename_axis([constants.GEO, constants.TIME])
          .to_frame()
          .to_xarray()
      )
      dataset = xr.combine_by_coords([dataset, revenue_per_kpi_xr])
    if self.coord_to_columns.media is not None:
      media_xr = (
          df_indexed[self.coord_to_columns.media]
          .stack()
          .rename(constants.MEDIA)
          .rename_axis(
              [constants.GEO, constants.MEDIA_TIME, constants.MEDIA_CHANNEL]
          )
          .to_frame()
          .to_xarray()
      )
      values = [x for x in media_xr.coords[constants.MEDIA_CHANNEL].values]

      print(f"values: '{values}'")
      first_value = values[0]
      print(f"first_value: {first_value}")
      print(f"media_to_channel: {self.media_to_channel}")
      print(f"media_to_channel[first_value]: {self.media_to_channel[first_value]}")
      media_xr.coords[constants.MEDIA_CHANNEL] = [
          self.media_to_channel[x]
          for x in media_xr.coords[constants.MEDIA_CHANNEL].values
      ]

      media_spend_xr = (
          df_indexed[self.coord_to_columns.media_spend]
          .stack()
          .rename(constants.MEDIA_SPEND)
          .rename_axis([constants.GEO, constants.TIME, constants.MEDIA_CHANNEL])
          .to_frame()
          .to_xarray()
      )
      media_spend_xr.coords[constants.MEDIA_CHANNEL] = [
          self.media_spend_to_channel[x]
          for x in media_spend_xr.coords[constants.MEDIA_CHANNEL].values
      ]
      dataset = xr.combine_by_coords([dataset, media_xr, media_spend_xr])

    if self.coord_to_columns.reach is not None:
      reach_xr = (
          df_indexed[self.coord_to_columns.reach]
          .stack()
          .rename(constants.REACH)
          .rename_axis(
              [constants.GEO, constants.MEDIA_TIME, constants.RF_CHANNEL]
          )
          .to_frame()
          .to_xarray()
      )
      reach_xr.coords[constants.RF_CHANNEL] = [
          self.reach_to_channel[x]
          for x in reach_xr.coords[constants.RF_CHANNEL].values
      ]

      frequency_xr = (
          df_indexed[self.coord_to_columns.frequency]
          .stack()
          .rename(constants.FREQUENCY)
          .rename_axis(
              [constants.GEO, constants.MEDIA_TIME, constants.RF_CHANNEL]
          )
          .to_frame()
          .to_xarray()
      )
      frequency_xr.coords[constants.RF_CHANNEL] = [
          self.frequency_to_channel[x]
          for x in frequency_xr.coords[constants.RF_CHANNEL].values
      ]

      rf_spend_xr = (
          df_indexed[self.coord_to_columns.rf_spend]
          .stack()
          .rename(constants.RF_SPEND)
          .rename_axis([constants.GEO, constants.TIME, constants.RF_CHANNEL])
          .to_frame()
          .to_xarray()
      )
      rf_spend_xr.coords[constants.RF_CHANNEL] = [
          self.rf_spend_to_channel[x]
          for x in rf_spend_xr.coords[constants.RF_CHANNEL].values
      ]
      dataset = xr.combine_by_coords(
          [dataset, reach_xr, frequency_xr, rf_spend_xr]
      )

    if self.coord_to_columns.organic_media is not None:
      organic_media_xr = (
          df_indexed[self.coord_to_columns.organic_media]
          .stack()
          .rename(constants.ORGANIC_MEDIA)
          .rename_axis([
              constants.GEO,
              constants.MEDIA_TIME,
              constants.ORGANIC_MEDIA_CHANNEL,
          ])
          .to_frame()
          .to_xarray()
      )
      dataset = xr.combine_by_coords([dataset, organic_media_xr])

    if self.coord_to_columns.organic_reach is not None:
      organic_reach_xr = (
          df_indexed[self.coord_to_columns.organic_reach]
          .stack()
          .rename(constants.ORGANIC_REACH)
          .rename_axis([
              constants.GEO,
              constants.MEDIA_TIME,
              constants.ORGANIC_RF_CHANNEL,
          ])
          .to_frame()
          .to_xarray()
      )
      organic_reach_xr.coords[constants.ORGANIC_RF_CHANNEL] = [
          self.organic_reach_to_channel[x]
          for x in organic_reach_xr.coords[constants.ORGANIC_RF_CHANNEL].values
      ]
      organic_frequency_xr = (
          df_indexed[self.coord_to_columns.organic_frequency]
          .stack()
          .rename(constants.ORGANIC_FREQUENCY)
          .rename_axis([
              constants.GEO,
              constants.MEDIA_TIME,
              constants.ORGANIC_RF_CHANNEL,
          ])
          .to_frame()
          .to_xarray()
      )
      organic_frequency_xr.coords[constants.ORGANIC_RF_CHANNEL] = [
          self.organic_frequency_to_channel[x]
          for x in organic_frequency_xr.coords[
              constants.ORGANIC_RF_CHANNEL
          ].values
      ]
      dataset = xr.combine_by_coords(
          [dataset, organic_reach_xr, organic_frequency_xr]
      )

    # Change back to geo names
    self.df[geo_column_name] = self.df[geo_column_name].replace(
        dict(zip(np.arange(len(geo_names)), geo_names))
    )
    dataset.coords[constants.GEO] = geo_names
    
    return load.XrDatasetDataLoader(dataset, kpi_type=self.kpi_type).load()