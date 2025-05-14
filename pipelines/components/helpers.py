import numpy as np
import pandas as pd
import os
import logging
import time
import datetime
from datetime import date
from typing import Optional

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


# --- Helper function for KPI Column Validation ---
def validate_kpi_column(df_to_validate: pd.DataFrame, kpi_col_name: str):
    """
    Validates the KPI column for numeric type, missing values, plausible range (non-negative),
    and sufficient variation.
    Modifies the DataFrame in-place if type conversion for the KPI column occurs.
    """
    if kpi_col_name not in df_to_validate.columns:
        logging.error(f"KPI column '{kpi_col_name}' not found in DataFrame. Skipping KPI validation.")
        return

    logging.info(f"--- Validating KPI Column: '{kpi_col_name}' ---")

    # 1. Numeric Data Type
    original_dtype = df_to_validate[kpi_col_name].dtype
    if not pd.api.types.is_numeric_dtype(df_to_validate[kpi_col_name]):
        logging.warning(f"KPI column '{kpi_col_name}' is not numeric (dtype: {original_dtype}). Attempting conversion.")
        # Modify a copy to check for introduced NaNs before assigning back
        converted_series = pd.to_numeric(df_to_validate[kpi_col_name], errors='coerce')
        original_nans = df_to_validate[kpi_col_name].isnull().sum()
        coerced_nans_total = converted_series.isnull().sum()
        newly_coerced_nans = coerced_nans_total - original_nans if coerced_nans_total >= original_nans else coerced_nans_total

        df_to_validate[kpi_col_name] = converted_series # Assign back to modify DataFrame in-place

        if pd.api.types.is_numeric_dtype(df_to_validate[kpi_col_name]):
            logging.info(f"KPI column '{kpi_col_name}' successfully converted to numeric. Newly coerced NaNs: {newly_coerced_nans}")
        else:
            logging.error(f"KPI column '{kpi_col_name}' could not be converted to a numeric type. Current dtype: {df_to_validate[kpi_col_name].dtype}.")
            # Not returning here, to allow other checks like missing values to still report if column exists
    else:
        logging.info(f"KPI column '{kpi_col_name}' is already numeric (dtype: {original_dtype}).")

    # 2. No Missing Values (after potential coercion)
    missing_values_count = df_to_validate[kpi_col_name].isnull().sum()
    if missing_values_count > 0:
        logging.warning(f"KPI column '{kpi_col_name}' has {missing_values_count} missing value(s).")
    else:
        logging.info(f"KPI column '{kpi_col_name}' has no missing values.")

    # Proceed with range and variation checks only if column is now numeric
    if pd.api.types.is_numeric_dtype(df_to_validate[kpi_col_name]):
        # 3. Plausible Range (e.g., non-negative)
        negative_values_mask = df_to_validate[kpi_col_name] < 0
        if negative_values_mask.any():
            negative_values_count = negative_values_mask.sum()
            logging.warning(f"KPI column '{kpi_col_name}' contains {negative_values_count} negative value(s).")
            logging.warning(f"Examples of negative KPI values: {df_to_validate.loc[negative_values_mask, kpi_col_name].head().tolist()}")
        else:
            logging.info(f"KPI column '{kpi_col_name}' does not contain negative values.")

        # 4. Sufficient Variation (on non-missing actual data)
        kpi_series_actual = df_to_validate[kpi_col_name].dropna()
        if kpi_series_actual.empty:
            logging.warning(f"KPI column '{kpi_col_name}' contains only missing values (or all values became NaN after coercion). No actual data to assess variation.")
        else:
            num_unique_values_actual = kpi_series_actual.nunique()
            std_dev_actual = kpi_series_actual.std()
            logging.info(f"KPI column '{kpi_col_name}' (actual data) has {num_unique_values_actual} unique value(s) and a standard deviation of {std_dev_actual:.4f}.")
            if num_unique_values_actual <= 1:
                logging.warning(f"KPI column '{kpi_col_name}' (actual data) has {num_unique_values_actual} unique value(s). This indicates insufficient/no variation for modeling.")
    else:
        logging.error(f"Cannot perform plausible range and variation checks as KPI column '{kpi_col_name}' is not numeric.")
    logging.info(f"--- KPI Column Validation for '{kpi_col_name}' Complete ---")
# --- End Helper function ---


# --- Helper function for Geo Column Validation ---
def validate_geo_column(df_to_validate: pd.DataFrame, geo_col_name: str, population_col_name: Optional[str] = None):
    """
    Validates the Geo/Region column for existence, missing values, consistent naming (by logging unique values),
    and alignment with population data (if provided).
    """
    if geo_col_name not in df_to_validate.columns:
        logging.error(f"Geo column '{geo_col_name}' not found in DataFrame. Skipping Geo validation.")
        return

    logging.info(f"--- Validating Geo Column: '{geo_col_name}' ---")

    # 1. No Missing Geo Values
    missing_geo_values_count = df_to_validate[geo_col_name].isnull().sum()
    if missing_geo_values_count > 0:
        logging.warning(f"Geo column '{geo_col_name}' has {missing_geo_values_count} missing value(s).")
    else:
        logging.info(f"Geo column '{geo_col_name}' has no missing values.")

    # 2. Consistent Naming/Coding (Log unique values and their counts for manual inspection)
    if not df_to_validate[geo_col_name].empty:
        unique_geos = df_to_validate[geo_col_name].value_counts(dropna=False) # include NaNs in counts if any
        logging.info(f"Unique values and their counts in geo column '{geo_col_name}':\n{unique_geos.to_string()}")
        if len(unique_geos) > 50: # Arbitrary threshold to warn about too many unique geos
            logging.warning(f"Geo column '{geo_col_name}' has a high number of unique values ({len(unique_geos)}). Review for consistency.")
    else:
        logging.info(f"Geo column '{geo_col_name}' is empty. No unique values to report.")

    # 3. Alignment with Population Data (if population_col_name is provided)
    if population_col_name:
        if population_col_name not in df_to_validate.columns:
            logging.warning(f"Population column '{population_col_name}' not found. Cannot check geo-population alignment.")
        else:
            # Check for rows where geo is present but population is missing
            mismatch_pop_df = df_to_validate[df_to_validate[geo_col_name].notnull() & df_to_validate[population_col_name].isnull()]
            if not mismatch_pop_df.empty:
                logging.warning(f"Found {len(mismatch_pop_df)} rows where geo ('{geo_col_name}') is present but population ('{population_col_name}') is missing.")
                logging.warning(f"First 5 examples of geo values with missing population:\n{mismatch_pop_df[[geo_col_name, population_col_name]].head().to_string()}")
            else:
                logging.info(f"Population data ('{population_col_name}') appears aligned with geo data ('{geo_col_name}') (no missing population for present geos).")
    logging.info(f"--- Geo Column Validation for '{geo_col_name}' Complete ---")
# --- End Helper function ---


# --- Helper function for Population Column Validation ---
def validate_population_column(df_to_validate: pd.DataFrame, population_col_name: str, geo_col_name: Optional[str] = None):
    """
    Validates the Population column for numeric type, plausible values (positive),
    and logs information about its consistency per geo.
    Modifies the DataFrame in-place if type conversion for the population column occurs.
    """
    if population_col_name not in df_to_validate.columns:
        logging.warning(f"Population column '{population_col_name}' not found in DataFrame. Skipping Population validation.")
        return

    logging.info(f"--- Validating Population Column: '{population_col_name}' ---")

    # 1. Numeric Data Type
    original_dtype = df_to_validate[population_col_name].dtype
    if not pd.api.types.is_numeric_dtype(df_to_validate[population_col_name]):
        logging.warning(f"Population column '{population_col_name}' is not numeric (dtype: {original_dtype}). Attempting conversion.")
        converted_series = pd.to_numeric(df_to_validate[population_col_name], errors='coerce')
        original_nans = df_to_validate[population_col_name].isnull().sum()
        coerced_nans_total = converted_series.isnull().sum()
        newly_coerced_nans = coerced_nans_total - original_nans if coerced_nans_total >= original_nans else coerced_nans_total

        df_to_validate[population_col_name] = converted_series # Assign back to modify DataFrame in-place

        if pd.api.types.is_numeric_dtype(df_to_validate[population_col_name]):
            logging.info(f"Population column '{population_col_name}' successfully converted to numeric. Newly coerced NaNs: {newly_coerced_nans}")
        else:
            logging.error(f"Population column '{population_col_name}' could not be converted to a numeric type. Current dtype: {df_to_validate[population_col_name].dtype}.")
            # Not returning here, to allow other checks like missing values to still report
    else:
        logging.info(f"Population column '{population_col_name}' is already numeric (dtype: {original_dtype}).")

    # Check for missing values (after potential coercion)
    missing_values_count = df_to_validate[population_col_name].isnull().sum()
    if missing_values_count > 0:
        logging.warning(f"Population column '{population_col_name}' has {missing_values_count} missing value(s).")
    else:
        logging.info(f"Population column '{population_col_name}' has no missing values.")

    # Proceed with plausible values and consistency checks only if column is now numeric
    if pd.api.types.is_numeric_dtype(df_to_validate[population_col_name]):
        # 2. Plausible Values (positive)
        non_positive_values_mask = df_to_validate[population_col_name] <= 0
        if non_positive_values_mask.any():
            non_positive_count = non_positive_values_mask.sum()
            logging.warning(f"Population column '{population_col_name}' contains {non_positive_count} zero or negative value(s).")
            logging.warning(f"Examples of non-positive population values: {df_to_validate.loc[non_positive_values_mask, population_col_name].head().tolist()}")
        else:
            logging.info(f"Population column '{population_col_name}' contains only positive values.")

        # 3. Consistency (Log unique population values per geo for manual inspection)
        if geo_col_name and geo_col_name in df_to_validate.columns:
            logging.info(f"Checking population consistency for '{population_col_name}' per geo ('{geo_col_name}'). Unique population counts per geo:")
            # Group by geo and count unique population values within each geo
            # This helps to see if population is static per geo or changes over time
            pop_consistency = df_to_validate.groupby(geo_col_name)[population_col_name].nunique()
            logging.info(f"\n{pop_consistency.to_string()}")
            if (pop_consistency > 1).any():
                logging.info("Some geos have varying population values over time. This may be expected or indicate an issue.")
            else:
                logging.info("Population values appear consistent (static) within each geo.")
    else:
        logging.error(f"Cannot perform plausible values and consistency checks as population column '{population_col_name}' is not numeric.")
    logging.info(f"--- Population Column Validation for '{population_col_name}' Complete ---")
# --- End Helper function ---


# --- Helper function for Control Variables Validation ---
def validate_control_variables(df_to_validate: pd.DataFrame, control_col_names: list[str]):
    """
    Validates control variable columns for appropriate data types, missing values,
    plausible ranges/categories, and sufficient variation.
    """
    if not control_col_names:
        logging.info("No control variables specified for validation.")
        return

    logging.info(f"--- Validating Control Variables: {control_col_names} ---")

    for col_name in control_col_names:
        if col_name not in df_to_validate.columns:
            logging.warning(f"Control variable column '{col_name}' not found in DataFrame. Skipping.")
            continue

        logging.info(f"  -- Validating Control Variable: '{col_name}' --")
        series_to_validate = df_to_validate[col_name]

        # 1. Data Type
        dtype = series_to_validate.dtype
        logging.info(f"  Data type of '{col_name}': {dtype}")

        # 2. Missing Values
        missing_values_count = series_to_validate.isnull().sum()
        if missing_values_count > 0:
            logging.warning(f"  Control variable '{col_name}' has {missing_values_count} missing value(s).")
        else:
            logging.info(f"  Control variable '{col_name}' has no missing values.")

        series_actual = series_to_validate.dropna()
        if series_actual.empty:
            logging.warning(f"  Control variable '{col_name}' contains only missing values. Cannot assess range or variation.")
            continue

        # 3. Plausible Ranges/Categories & 4. Sufficient Variation
        num_unique_values_actual = series_actual.nunique()
        logging.info(f"  Control variable '{col_name}' (actual data) has {num_unique_values_actual} unique value(s).")

        if pd.api.types.is_numeric_dtype(series_actual):
            # For numeric control variables
            logging.info(f"    Min: {series_actual.min()}, Max: {series_actual.max()}, Mean: {series_actual.mean():.4f}, StdDev: {series_actual.std():.4f}")
            if num_unique_values_actual <= 1:
                logging.warning(f"    Numeric control variable '{col_name}' has insufficient variation (<=1 unique value).")
            elif series_actual.std() == 0:
                 logging.warning(f"    Numeric control variable '{col_name}' has zero standard deviation, indicating no variation.")
        else:
            # For categorical/boolean or other non-numeric control variables
            unique_value_counts = series_actual.value_counts(dropna=False).sort_index()
            logging.info(f"    Unique value counts for '{col_name}':\n{unique_value_counts.to_string()}")
            if num_unique_values_actual <= 1:
                logging.warning(f"    Categorical/Object control variable '{col_name}' has insufficient variation (<=1 unique value).")

    logging.info(f"--- Control Variables Validation Complete ---")
# --- End Helper function ---


# --- Helper function for Media Spend/Impression/Cost Columns Validation ---
def validate_media_columns(
    df_to_validate: pd.DataFrame,
    media_impression_col_names: list[str],
    media_spend_col_names: list[str],
    media_to_channel_map: dict[str, str] # For logging canonical channel names
):
    """
    Validates media impression and spend columns for data types, non-negativity,
    missing values, plausible ranges, consistency, and implied cost per unit.
    Modifies the DataFrame in-place if type conversions occur.
    """
    if not media_impression_col_names and not media_spend_col_names:
        logging.info("No media impression or spend columns specified for validation.")
        return

    logging.info(f"--- Validating Media Columns ---")

    if len(media_impression_col_names) != len(media_spend_col_names):
        logging.error("Mismatch in length between media impression and spend column lists. Cannot perform paired validation accurately. Please check coord_to_columns.")
        # Proceed with individual column checks if possible, but skip paired checks.
        # For simplicity in this example, we'll primarily focus on the paired iteration.
        # A more robust implementation might handle them separately if lengths differ.

    num_channels = len(media_impression_col_names)

    for i in range(num_channels):
        impression_col = media_impression_col_names[i] if i < len(media_impression_col_names) else None
        spend_col = media_spend_col_names[i] if i < len(media_spend_col_names) else None
        
        # Determine canonical channel name for logging (prefer impression, fallback to spend if impression_col is None)
        canonical_channel_name = "Unknown_Channel"
        if impression_col and impression_col in media_to_channel_map:
            canonical_channel_name = media_to_channel_map.get(impression_col, f"Channel_Imp_{i}")
        elif spend_col : # Try to get from spend_col if impression_col is not available or not in map
             # Assuming a similar spend_to_channel_map exists or can be inferred
             # For this example, we'll use a placeholder if not directly available
             # In a real scenario, you'd pass media_spend_to_channel_map as well
            canonical_channel_name = f"Channel_Spd_{i}" # Placeholder

        logging.info(f"  -- Validating Media for Channel: '{canonical_channel_name}' (Imp: {impression_col}, Spend: {spend_col}) --")

        # Validate Impression Column
        if impression_col:
            if impression_col not in df_to_validate.columns:
                logging.warning(f"    Impression column '{impression_col}' not found. Skipping its validation.")
            else:
                series_imp = df_to_validate[impression_col]
                # Numeric Type
                if not pd.api.types.is_numeric_dtype(series_imp):
                    logging.warning(f"    Impression column '{impression_col}' is not numeric (dtype: {series_imp.dtype}). Attempting conversion.")
                    df_to_validate[impression_col] = pd.to_numeric(series_imp, errors='coerce')
                    if not pd.api.types.is_numeric_dtype(df_to_validate[impression_col]):
                         logging.error(f"    Could not convert impression column '{impression_col}' to numeric.")
                # Update series_imp after potential conversion
                series_imp = df_to_validate[impression_col]
                
                if pd.api.types.is_numeric_dtype(series_imp):
                    # Missing Values (NaN)
                    missing_imp = series_imp.isnull().sum()
                    if missing_imp > 0: logging.warning(f"    Impression column '{impression_col}' has {missing_imp} NaN value(s).")
                    # Zero Values
                    zeros_imp = (series_imp == 0).sum()
                    if zeros_imp > 0: logging.info(f"    Impression column '{impression_col}' has {zeros_imp} zero value(s).")
                    # Non-Negative
                    negative_imp = (series_imp < 0).sum()
                    if negative_imp > 0: logging.warning(f"    Impression column '{impression_col}' has {negative_imp} negative value(s).")
                    # Plausible Range (on actual, non-negative data)
                    actual_imp = series_imp.dropna()[series_imp.dropna() >= 0]
                    if not actual_imp.empty: logging.info(f"    Impression '{impression_col}' (actual non-negative): Min={actual_imp.min()}, Max={actual_imp.max()}, Mean={actual_imp.mean():.2f}, Std={actual_imp.std():.2f}")
                    else: logging.info(f"    Impression '{impression_col}' has no non-negative actual data to analyze range.")

        # Validate Spend Column
        if spend_col:
            if spend_col not in df_to_validate.columns:
                logging.warning(f"    Spend column '{spend_col}' not found. Skipping its validation.")
            else:
                series_spend = df_to_validate[spend_col]
                # Numeric Type
                if not pd.api.types.is_numeric_dtype(series_spend):
                    logging.warning(f"    Spend column '{spend_col}' is not numeric (dtype: {series_spend.dtype}). Attempting conversion.")
                    df_to_validate[spend_col] = pd.to_numeric(series_spend, errors='coerce')
                    if not pd.api.types.is_numeric_dtype(df_to_validate[spend_col]):
                        logging.error(f"    Could not convert spend column '{spend_col}' to numeric.")
                # Update series_spend after potential conversion
                series_spend = df_to_validate[spend_col]

                if pd.api.types.is_numeric_dtype(series_spend):
                    # Missing Values (NaN)
                    missing_spend = series_spend.isnull().sum()
                    if missing_spend > 0: logging.warning(f"    Spend column '{spend_col}' has {missing_spend} NaN value(s).")
                    # Zero Values
                    zeros_spend = (series_spend == 0).sum()
                    if zeros_spend > 0: logging.info(f"    Spend column '{spend_col}' has {zeros_spend} zero value(s).")
                    # Non-Negative
                    negative_spend = (series_spend < 0).sum()
                    if negative_spend > 0: logging.warning(f"    Spend column '{spend_col}' has {negative_spend} negative value(s).")
                    # Plausible Range (on actual, non-negative data)
                    actual_spend = series_spend.dropna()[series_spend.dropna() >= 0]
                    if not actual_spend.empty: logging.info(f"    Spend '{spend_col}' (actual non-negative): Min={actual_spend.min()}, Max={actual_spend.max()}, Mean={actual_spend.mean():.2f}, Std={actual_spend.std():.2f}")
                    else: logging.info(f"    Spend '{spend_col}' has no non-negative actual data to analyze range.")

        # Consistency and Cost per Unit (if both columns are valid and numeric)
        if impression_col and spend_col and \
           impression_col in df_to_validate.columns and spend_col in df_to_validate.columns and \
           pd.api.types.is_numeric_dtype(df_to_validate[impression_col]) and \
           pd.api.types.is_numeric_dtype(df_to_validate[spend_col]):
            
            df_subset = df_to_validate[[impression_col, spend_col]].copy() # Work on a copy for safety
            
            spend_gt_0_imp_eq_0 = df_subset[(df_subset[spend_col] > 0) & (df_subset[impression_col] == 0)].shape[0]
            if spend_gt_0_imp_eq_0 > 0: logging.warning(f"    Found {spend_gt_0_imp_eq_0} instances where spend > 0 and impressions == 0 for channel '{canonical_channel_name}'.")

            imp_gt_0_spend_eq_0 = df_subset[(df_subset[impression_col] > 0) & (df_subset[spend_col] == 0)].shape[0]
            if imp_gt_0_spend_eq_0 > 0: logging.warning(f"    Found {imp_gt_0_spend_eq_0} instances where impressions > 0 and spend == 0 for channel '{canonical_channel_name}'.")

            # Cost per Impression (CPI)
            # Calculate only where impressions > 0 to avoid division by zero
            cpi_series = df_subset[df_subset[impression_col] > 0][spend_col] / df_subset[df_subset[impression_col] > 0][impression_col]
            cpi_series = cpi_series.dropna() # Remove NaNs that might result from spend being NaN
            if not cpi_series.empty:
                logging.info(f"    Cost Per Impression (CPI) for '{canonical_channel_name}': Min={cpi_series.min():.4f}, Max={cpi_series.max():.4f}, Mean={cpi_series.mean():.4f}, Std={cpi_series.std():.4f}")
            else:
                logging.info(f"    Could not calculate CPI for '{canonical_channel_name}' (e.g., no impressions > 0 or spend data missing).")

    logging.info(f"--- Media Columns Validation Complete ---")
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
# --- End Helper function ---





def reload_dataframe(self) -> input_data.InputData:
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
