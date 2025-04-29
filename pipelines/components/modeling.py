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

from kfp.dsl import component, Output, Model, Dataset
import os
import yaml

config_file_path = os.path.join(os.path.dirname(
    __file__), '../../config/config.yaml')

base_image = None
if os.path.exists(config_file_path):
    with open(config_file_path, encoding='utf-8') as fh:
        configs = yaml.full_load(fh)

    vertex_components_params = configs['vertex_ai']['components']
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

PROJECT_ID = "meridian-dev-455515"
LOCATION = "us-central1"
BUCKET_NAME ="meridian-dev-455515-pipelines"
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

GPU_TRAIN_IMAGE = "us-central1-docker.pkg.dev/meridian-dev-455515/pipelines-docker-repo/meridian-gpu-base-image:dev"
CPU_TRAIN_IMAGE = "us-central1-docker.pkg.dev/meridian-dev-455515/pipelines-docker-repo/meridian-cpu-base-image:dev"

MACHINE_TYPE = "n1-standard"

VCPU = "4"
TRAIN_COMPUTE = MACHINE_TYPE + "-" + VCPU

PIPELINE_ROOT = "{}/pipeline_root/machine_settings".format(BUCKET_URI)

@dsl.component(
    base_image=GPU_TRAIN_IMAGE,
)
def train_meridian_model(
    project_id: str,
    bq_dataset: str,
    bq_table_name: str,
    roi_mu: float, roi_sigma: float, n_chains: int,
    n_adapt: int, n_burnin: int, n_keep: int, seed: int,
    output_model: Output[Model],
):
    # --- Imports inside component ---
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import tensorflow_probability as tfp
    import os
    import logging
    import time
    import datetime
    from google.cloud import bigquery
    from meridian import constants
    from meridian.data import load
    from meridian.model import model, spec, prior_distribution
    import dill

    # --- Reconfigure logging inside component if needed, or rely on root config ---
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Optional reconfig

    MERIDIAN_MODEL_FILENAME = "model_save.pkl" # Model Name

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
        time='time', geo='geo', controls=['GQV', 'Competitor_Sales'], population='population',
        kpi='conversions', revenue_per_kpi='revenue_per_conversion',
        media=[f'Channel{i}_impression' for i in range(5)], ## HERE FOR THE SAMPLE DATASET, change with your own channels names
        media_spend=[f'Channel{i}_spend' for i in range(5)], ## HERE FOR THE SAMPLE DATASET, change with your own channels names
        organic_media=['Organic_channel0_impression'], non_media_treatments=['Promo'],
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

        # --- Convert time column, BQ_To_Dataframe converts the datetime so we need to convert it yyyy-mm-dd ---
        time_col_name = coord_to_columns.time
        if time_col_name in df.columns:
            logging.info(f"Converting time column '{time_col_name}' to string format 'YYYY-MM-DD'")
            if pd.api.types.is_datetime64_any_dtype(df[time_col_name]) or isinstance(df[time_col_name].iloc[0], pd.Timestamp) or isinstance(df[time_col_name].iloc[0], datetime.date):
                 df[time_col_name] = pd.to_datetime(df[time_col_name]).dt.strftime('%Y-%m-%d')
                 logging.info(f"Conversion of '{time_col_name}' complete.")
            elif pd.api.types.is_string_dtype(df[time_col_name]):
                 logging.info(f"Column '{time_col_name}' is already string type. Checking format (first row): {df[time_col_name].iloc[0]}")
            else:
                 logging.warning(f"Column '{time_col_name}' is not a recognized datetime or string type ({df[time_col_name].dtype}). Meridian might still fail.")
        else:
            logging.error(f"Specified time column '{time_col_name}' not found in DataFrame!")
            raise ValueError(f"Time column '{time_col_name}' defined in coord_to_columns not found in BigQuery results.")
        # --- End Time Conversion ---

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


    logging.info("Configuring model...")
    prior = prior_distribution.PriorDistribution(
        roi_m=tfp.distributions.LogNormal(roi_mu, roi_sigma, name=constants.ROI_M)
    )
    model_spec_obj = spec.ModelSpec(prior=prior)
    mmm = model.Meridian(input_data=data, model_spec=model_spec_obj) # Use the 'data' object loaded from BQ

    logging.info("Sampling prior...")
    mmm.sample_prior(500)
    logging.info(f"Sampling posterior with {n_chains} chains...")
    start_time = time.time()
    mmm.sample_posterior(
        n_chains=n_chains, n_adapt=n_adapt, n_burnin=n_burnin, n_keep=n_keep, seed=seed
    )
    end_time = time.time()
    logging.info(f"Posterior sampling complete. Duration: {end_time - start_time:.2f} seconds.")

    save_file_path = os.path.join(output_model.path, MERIDIAN_MODEL_FILENAME)
    logging.info(f"Saving model artifact using model.save_mmm to file: {save_file_path}")
    try:
        os.makedirs(output_model.path, exist_ok=True)
        model.save_mmm(mmm, save_file_path)
        logging.info("Model saved successfully using meridian.model.model.save_mmm.")
    except Exception as e:
        logging.error(f"meridian.model.model.save_mmm failed: {e}")
        raise e

    output_model.metadata["framework"] = "Meridian"
    output_model.metadata["saved_filename"] = MERIDIAN_MODEL_FILENAME
    output_model.metadata["description"] = f"Trained Meridian MMM model (BQ Input, saved via save_mmm to {MERIDIAN_MODEL_FILENAME})"
    logging.info("Training component finished.")
    return


@dsl.component(
    base_image=CPU_TRAIN_IMAGE,
)
def generate_summary_report(
    model_artifact: Input[Model],
    output_gcs_dir: str,
    report_filename: str,
    start_date: str,
    end_date: str,
    summary_report_artifact: Output[Artifact],
):
    import os
    import logging
    import time
    import tempfile
    from meridian.analysis import summarizer # Use summarizer for HTML report
    from meridian.model import model
    from google.cloud import storage
    from urllib.parse import urlparse
    import dill # Ensure dill is imported, needed by load_mmm

    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Optional reconfig
    MERIDIAN_MODEL_FILENAME = "model_save.pkl"
    def upload_local_file_to_gcs(local_path: str, gcs_uri: str):
        storage_client = storage.Client()
        parsed_uri = urlparse(gcs_uri)
        bucket_name = parsed_uri.netloc
        destination_blob_name = parsed_uri.path.lstrip('/')
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(local_path)
        logging.info(f"File {local_path} uploaded to {gcs_uri}")

    model_dir_path = model_artifact.path
    load_file_path = os.path.join(model_dir_path, MERIDIAN_MODEL_FILENAME)
    logging.info(f"Attempting to load model from file: {load_file_path}")
    if not os.path.exists(load_file_path):
        raise FileNotFoundError(f"Expected model file {MERIDIAN_MODEL_FILENAME} not found in {model_dir_path}")
    try:
        mmm = model.load_mmm(load_file_path)
        logging.info("Model loaded successfully for HTML report generation.")
    except Exception as e:
        logging.error(f"Model loading failed: {e}")
        raise e

    if not output_gcs_dir.startswith("gs://"):
        raise ValueError("output_gcs_dir must be a GCS path (gs://...)")
    final_gcs_uri = os.path.join(output_gcs_dir, report_filename)

    with tempfile.TemporaryDirectory() as temp_dir:
        logging.info(f"Generating summary HTML report locally in: {temp_dir}")
        local_report_source_path = os.path.join(temp_dir, report_filename)
        try:
            # Use Summarizer for the HTML report as in original code
            mmm_summarizer = summarizer.Summarizer(mmm)
            mmm_summarizer.output_model_results_summary(
                filename=report_filename,
                filepath=temp_dir,
                start_date=start_date,
                end_date=end_date
            )
            logging.info(f"Meridian saved HTML report locally to: {local_report_source_path}")
            if not os.path.exists(local_report_source_path):
                logging.error(f"Meridian did not create the expected local HTML report file: {local_report_source_path}")
                raise FileNotFoundError(f"HTML Report file not created locally by Meridian at {local_report_source_path}")
            logging.info(f"Manually uploading {local_report_source_path} to {final_gcs_uri}")
            upload_local_file_to_gcs(local_report_source_path, final_gcs_uri)
            summary_report_artifact.uri = final_gcs_uri
            summary_report_artifact.metadata["gcs_path"] = final_gcs_uri
            summary_report_artifact.metadata["filename"] = report_filename
            logging.info(f"Set KFP artifact URI for HTML report to: {summary_report_artifact.uri}")
        except Exception as e:
            logging.error(f"Failed to generate or upload HTML summary report: {e}")
            raise e
    logging.info("HTML Summary report component finished.")
    return


# --- Generate and Save Summary Table to BigQuery ---
@dsl.component(
    base_image=CPU_TRAIN_IMAGE,
)
def generate_and_save_summary_bq(
    model_artifact: Input[Model],
    project_id: str,
    bq_dataset: str,
    bq_table_name: str, # Target table for this summary
    bq_output_table: Output[Artifact], # Output artifact to track the BQ table
):
    import os
    import logging
    import pandas_gbq
    import pandas as pd
    from meridian.analysis import visualizer # Use visualizer as per user image for the table
    from meridian.model import model
    from google.cloud import bigquery
    import dill # Ensure dill is imported if needed by load_mmm

    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Optional reconfig
    MERIDIAN_MODEL_FILENAME = "model_save.pkl"

    model_dir_path = model_artifact.path
    load_file_path = os.path.join(model_dir_path, MERIDIAN_MODEL_FILENAME)
    logging.info(f"Attempting to load model from file: {load_file_path} for BQ summary")
    if not os.path.exists(load_file_path):
        raise FileNotFoundError(f"Expected model file {MERIDIAN_MODEL_FILENAME} not found in {model_dir_path}")

    try:
        mmm = model.load_mmm(load_file_path)
        logging.info("Model loaded successfully for BQ summary generation.")
    except Exception as e:
        logging.error(f"Model loading failed: {e}")
        raise e

    logging.info("Generating media summary table using visualizer.MediaSummary...")
    try:
        # Instantiate the visualizer's MediaSummary class
        media_summary_visualizer = visualizer.MediaSummary(mmm)
        summary_df = media_summary_visualizer.summary_table()
        logging.info("Successfully generated summary DataFrame.")
        logging.info("First 5 rows of summary DataFrame:")
        logging.info(summary_df.head().to_string())
        logging.info("\nDataFrame Info:")
        # Use a StringIO buffer to capture info() output for logging
        import io
        buffer = io.StringIO()
        summary_df.info(buf=buffer)
        logging.info(buffer.getvalue())

    except AttributeError:
         logging.error("AttributeError: Could not find 'MediaSummary' or 'summary_table' in 'meridian.analysis.visualizer'. "
                       "Perhaps the class/method name is different or in another module (e.g., summarizer)?")
         # --- Fallback attempt using Summarizer if Visualizer fails ---
         logging.warning("Attempting fallback using meridian.analysis.summarizer.Summarizer...")
         try:
             from meridian.analysis import summarizer
             mmm_summarizer = summarizer.Summarizer(mmm)
             if hasattr(mmm_summarizer, 'get_summary_dataframe'):
                 summary_df = mmm_summarizer.get_summary_dataframe() # Hypothetical method
                 logging.info("Successfully generated summary DataFrame using Summarizer fallback.")
             elif hasattr(mmm_summarizer, '_create_summary_table'): # Check private methods if desperate
                  summary_df = mmm_summarizer._create_summary_table() # Highly discouraged, likely to break
                  logging.info("Successfully generated summary DataFrame using Summarizer fallback (_create_summary_table).")
             else:
                 logging.error("Fallback failed: Summarizer does not have a known method to return the summary DataFrame.")
                 raise ValueError("Could not generate summary DataFrame using known Meridian methods.")
         except Exception as fallback_e:
             logging.error(f"Error during Summarizer fallback: {fallback_e}")
             raise fallback_e # Re-raise the fallback error
         # --- End Fallback attempt ---
    except Exception as e:
        logging.error(f"Failed to generate summary table: {e}")
        raise e

    # --- Prepare DataFrame for BigQuery ---
    # BQ prefers snake_case column names without special characters or spaces
    original_columns = summary_df.columns.tolist()
    new_columns = []
    for col in original_columns:
        new_col = str(col).lower() # Convert to string just in case, then lowercase
        new_col = new_col.replace('% ', 'pct_').replace(' ', '_').replace('.', '').replace('(', '').replace(')', '')
        new_columns.append(new_col)
    summary_df.columns = new_columns
    logging.info(f"Renamed DataFrame columns for BQ compatibility: {new_columns}")

    # Convert complex object columns (like tuples represented as strings) to plain strings
    # This prevents potential 'to_gbq' errors with complex types
    for col in summary_df.columns:
        if summary_df[col].dtype == 'object':
            # Check if the first non-null value looks like a tuple/list string representation
            first_val = summary_df[col].dropna().iloc[0] if not summary_df[col].dropna().empty else None
            if isinstance(first_val, (tuple, list)) or (isinstance(first_val, str) and first_val.strip().startswith(('(', '['))):
                 logging.info(f"Converting object column '{col}' to string for BQ.")
                 summary_df[col] = summary_df[col].astype(str)
            elif pd.api.types.is_numeric_dtype(summary_df[col].dropna()):
                # Sometimes mixed types get 'object', try converting back to numeric if possible
                 try:
                     summary_df[col] = pd.to_numeric(summary_df[col])
                     logging.info(f"Converted object column '{col}' back to numeric.")
                 except: # Keep as object/string if conversion fails
                     logging.warning(f"Could not convert object column '{col}' to numeric, keeping as object/string.")
                     summary_df[col] = summary_df[col].astype(str) # Ensure string if not numeric
            else: # Default to string conversion for other objects
                 logging.info(f"Converting object column '{col}' to string for BQ.")
                 summary_df[col] = summary_df[col].astype(str)


    # Handle potential 'nan' strings from conversions if needed
    summary_df = summary_df.fillna(pd.NA).replace(['nan', 'NaN', 'None', '(nan, nan)', 'nan (nan, nan)'], [pd.NA, pd.NA, pd.NA, pd.NA, pd.NA]) # Replace various nan strings with proper NA for BQ

    # Reset index if it's meaningful (like the 0, 1, 2... row numbers) to make it a column
    if summary_df.index.name is None and pd.api.types.is_integer_dtype(summary_df.index):
         summary_df = summary_df.reset_index()
         # Rename the new 'index' column if desired
         index_col_name = 'original_index'
         if index_col_name in summary_df.columns: # Avoid collision
             index_col_name = 'row_index'
         summary_df = summary_df.rename(columns={'index': index_col_name})
         logging.info(f"Reset DataFrame index and added column '{index_col_name}'.")

    logging.info("Final DataFrame Schema before BQ Upload:")
    buffer = io.StringIO()
    summary_df.info(buf=buffer)
    logging.info(buffer.getvalue())
    logging.info("First 5 rows before BQ Upload:")
    logging.info(summary_df.head().to_string())


    # --- Save to BigQuery ---
    bq_table_full_id = f"{project_id}.{bq_dataset}.{bq_table_name}"
    logging.info(f"Attempting to save summary DataFrame to BigQuery table: {bq_table_full_id}")

    try:
        client = bigquery.Client(project=project_id)
        logging.info("BigQuery client created successfully.")

        # Use pandas_gbq or DataFrame.to_gbq (uses pandas_gbq backend)
        summary_df.to_gbq(
            destination_table=f"{bq_dataset}.{bq_table_name}",
            project_id=project_id,
            if_exists='replace', # Options: 'fail', 'replace', 'append'
            # Optional: Define schema explicitly for more control if needed
            # table_schema=[{'name': 'col1', 'type': 'STRING'}, ...]
        )
        logging.info(f"Successfully wrote summary data to BigQuery table: {bq_table_full_id}")

        # Set output artifact metadata
        bq_output_table.metadata["table_id"] = bq_table_full_id
        bq_output_table.uri = f"https://console.cloud.google.com/bigquery?project={project_id}&ws=!1m5!1m4!4m3!1s{project_id}!2s{bq_dataset}!3s{bq_table_name}" # URI to the BQ table

    except Exception as e:
        logging.error(f"Failed to write DataFrame to BigQuery: {e}")
        # Log dataframe details that might cause issues
        logging.error(f"DataFrame dtypes:\n{summary_df.dtypes}")
        raise e

    logging.info("Generate and Save Summary to BQ component finished.")
    return
