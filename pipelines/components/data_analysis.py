# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Optional

from helpers import convert_time_column_to_string, detect_and_log_time_outliers, check_and_log_zero_negative_values


from kfp.dsl import component, Output, Model, Dataset
import os
import yaml

config_file_path = os.path.join(os.path.dirname(
    __file__), '../../config/config.yaml')

base_image = None
repo_params = None
vertex_components_params = None
vertex_pipelines_params = None
if os.path.exists(config_file_path):
    with open(config_file_path, encoding='utf-8') as fh:
        configs = yaml.full_load(fh)

    vertex_components_params = configs['vertex_ai']['components']
    vertex_pipelines_params = configs['vertex_ai']['pipelines']
    repo_params = configs['artifact_registry']['pipelines_docker_repo']

    # defines the base_image variable, which specifies the Docker image to be used for the component. This image is retrieved from the config.yaml file, which contains configuration parameters for the project.
    base_image = f"{repo_params['region']}-docker.pkg.dev/{repo_params['project_id']}/{repo_params['name']}/{vertex_components_params['base_image_name']}:{vertex_components_params['base_image_tag']}"
    gpu_base_image = f"{repo_params['region']}-docker.pkg.dev/{repo_params['project_id']}/{repo_params['name']}/{vertex_components_params['gpu_base_image_name']}:{vertex_components_params['gpu_base_image_tag']}"


import json

import google.cloud.aiplatform as aiplatform
import vertexai

from google_cloud_pipeline_components.v1.custom_job import utils
from google_cloud_pipeline_components.preview.custom_job.component import custom_training_job

from kfp import compiler, dsl
import kfp
from kfp.dsl import Artifact, Dataset, Input, Metrics, Model, Output, component

PROJECT_ID = repo_params['project_id']
LOCATION = repo_params['region']
BUCKET_NAME = vertex_pipelines_params['bucket_name']
BUCKET_URI = f"gs://{BUCKET_NAME}"

# Get the OAuth2 token.
# Once you've obtained the OAuth2 token, use it to make an authenticated call
# to the target audience.
import google.auth
from google.auth import impersonated_credentials
import google.auth.transport.requests

credentials, _ = google.auth.default()
request = google.auth.transport.requests.Request()
credentials.refresh(request)
credentials.apply(headers = {'user-agent': 'cloud-solutions/mas-meridian-on-gcp-usage-v1.0'})
credentials.refresh(request)
if credentials.valid:
  print('Authenticated')

aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=BUCKET_URI)

TRAIN_GPU, TRAIN_NGPU = (aiplatform.gapic.AcceleratorType.NVIDIA_TESLA_T4, 1)
DEPLOY_GPU, DEPLOY_NGPU = (None, None)

GPU_TRAIN_IMAGE = gpu_base_image
CPU_TRAIN_IMAGE = base_image

MACHINE_TYPE = "n1-standard"

VCPU = "4"
TRAIN_COMPUTE = MACHINE_TYPE + "-" + VCPU

PIPELINE_ROOT = "{}/pipeline_root/machine_settings".format(BUCKET_URI)

@component(
    base_image=CPU_TRAIN_IMAGE,
)
def meridian_data_analysis_component(
    project_id: str,
    bq_dataset: str,
    bq_table_name: str,
    reload_data: bool = False,
) -> str:

    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import tensorflow_probability as tfp
    import os
    import logging
    import time
    import datetime
    from datetime import date
    from google.cloud import bigquery
    from meridian import constants
    from meridian.data import load
    from meridian.model import model, spec, prior_distribution
    import dill

    # --- Reconfigure logging inside component if needed, or rely on root config ---
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Optional reconfig

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logging.info(f"GPUs available: {gpus}")
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
            logging.info("Enabled memory growth for GPUs.")
        except RuntimeError as e: logging.error(f"Error setting memory growth: {e}")
    else: logging.warning("No GPU detected by TensorFlow. Running on CPU.")

    # --- Define Mappings ---
    coord_to_columns = load.CoordToColumns(
        time='time', 
        geo='geo', 
        controls=['GQV', 'Competitor_Sales'], 
        population='population',
        kpi='conversions', 
        revenue_per_kpi='revenue_per_conversion',
        media=[f'Channel{i}_impression' for i in range(5)], ## HERE FOR THE SAMPLE DATASET, change with your own channels names
        media_spend=[f'Channel{i}_spend' for i in range(5)], ## HERE FOR THE SAMPLE DATASET, change with your own channels names
        organic_media=['Organic_channel0_impression'], 
        non_media_treatments=['Promo'],
    )
    correct_media_to_channel = {f'Channel{i}_impression': f'Channel_{i}' for i in range(5)} ## HERE FOR THE SAMPLE DATASET, change with your own channels names
    correct_media_spend_to_channel = {f'Channel{i}_spend': f'Channel_{i}' for i in range(5)} ## HERE FOR THE SAMPLE DATASET, change with your own channels names
    # ----------------------------------------------------------------------

    # --- BigQuery Data Loading Start ---
    bq_table_full_id = f"{project_id}.{bq_dataset}.{bq_table_name}"
    logging.info(f"Attempting to load data from BigQuery table: {bq_table_full_id}")

    try:
        client = bigquery.Client(project=project_id)
        logging.info("BigQuery client created successfully.")
    except Exception as e:
        logging.error(f"Failed to create BigQuery client: {e}")
        raise e

    sql_query = f"SELECT * FROM `{bq_table_full_id}`"
    logging.info(f"Executing query: {sql_query}")

    try:
        df = client.query(sql_query).to_dataframe()
        logging.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns from BigQuery.")

        time_col_name = coord_to_columns.time
        df = convert_time_column_to_string(df, time_col_name)

        # Define expected date range and call the outlier detection function
        min_expected_date_str = "2000-01-01"
        max_expected_date_str = date.today().strftime('%Y-%m-%d')
        detect_and_log_time_outliers(df, time_col_name, min_expected_date_str, max_expected_date_str)
        
        # --- Perform Non-Negative Value Checks ---
        columns_to_check_non_negative = []
        # KPI
        columns_to_check_non_negative.append(df.columns.kpi)
        # Revenue per KPI
        columns_to_check_non_negative.append(df.columns.revenue_per_kpi)
        # Media Spend
        for col in coord_to_columns.media_spend:
            if col in df.columns:
                columns_to_check_non_negative.append(col)
        # Organic Media
        for col in coord_to_columns.organic_media:
            if col in df.columns:
                columns_to_check_non_negative.append(col)
        
        if columns_to_check_non_negative:
            check_and_log_zero_negative_values(df, columns_to_check_non_negative)
        # --- End Non-Negative Value Checks ---
        
        if reload_data:
            load.DataFrameDataLoader.load = new_load

        logging.info("First 5 rows of loaded data (post-conversion):")
        logging.info(df.head().to_string()) # Use to_string for logging DataFrames

    except Exception as e:
        logging.error(f"Error loading data from BigQuery or processing DataFrame: {e}")
        raise e

    # --- Use DataFrameDataLoader ---
    logging.info("Initializing Meridian DataFrameDataLoader...")
    try:
        loader = load.DataFrameDataLoader(
            df=df, # Pass the DataFrame loaded from BQ
            kpi_type='non_revenue',
            coord_to_columns=coord_to_columns,
            media_to_channel=correct_media_to_channel,
            media_spend_to_channel=correct_media_spend_to_channel,
        )
        data = loader.load()
        logging.info("Data successfully loaded into Meridian InputData format.")
    except Exception as e:
        logging.error(f"Error during Meridian data loading process (DataFrameDataLoader): {e}")
        raise e
    # --- BigQuery Data Loading End ---

    return