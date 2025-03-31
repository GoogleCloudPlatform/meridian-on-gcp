#Copyright 2024 Google LLC
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import kfp
from kfp.v2 import compiler, dsl
from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Artifact,
    Dataset,
    Model
)
from google.cloud import aiplatform
from google_cloud_pipeline_components import aiplatform as gcc_aip
from typing import NamedTuple

# Define pipeline parameters
PROJECT_ID = "your-project-id"  # Replace with your project ID
REGION = "us-central1"  # Replace with your desired region
PIPELINE_ROOT = "gs://your-bucket/pipeline_root"  # Replace with your GCS bucket
BQ_DATASET = "your_bq_dataset"   # Replace with your BQ dataset
BQ_TABLE_IN = "your_input_bq_table"  # Replace with your input BQ table
BQ_TABLE_OUT = "your_output_bq_table"  # Replace with your output BQ table
GCS_OUTPUT_DIR = "gs://your-bucket/meridian_output"  # Replace with your GCS output directory
TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
BASE_IMAGE = "python:3.9" # or a custom image with required dependencies
DISPLAY_NAME = "meridian-pipeline-job-" + TIMESTAMP

# 1.  Data Loading Component (from BigQuery)
@component(base_image=BASE_IMAGE, packages_to_install=["google-cloud-bigquery", "pandas", "db_dtypes"])
def load_data_from_bq(
    project_id: str,
    bq_dataset: str,
    bq_table: str,
    output_data: Output[Dataset],
):
    """Loads data from a BigQuery table and saves it to a Dataset artifact."""
    from google.cloud import bigquery
    import pandas as pd

    client = bigquery.Client(project=project_id)

    query = f"""
        SELECT *
        FROM `{project_id}.{bq_dataset}.{bq_table}`
    """
    df = client.query(query).to_dataframe()
    df.to_csv(output_data.path + ".csv", index=False)  # Save as CSV
    print(f"Data loaded from {project_id}.{bq_dataset}.{bq_table} and saved as CSV.")

    # Add metadata (very important for lineage tracking!)
    output_data.metadata["source"] = f"bq://{project_id}.{bq_dataset}.{bq_table}"
    output_data.metadata["format"] = "csv"

# 2. Meridian Processing Component
@component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        "google-meridian[colab,and-cuda]",
        "numpy",
        "pandas",
        "tensorflow",
        "tensorflow-probability",
        "arviz",
        "psutil",
        "db_dtypes",  #for dataframe.to_gbq
        'google-cloud-storage' #for bigquery to save

    ],
)
def run_meridian_analysis(
    input_data: Input[Dataset],
    project_id: str,
    bq_dataset: str,
    output_bq_table: str,
    gcs_output_dir: str,
    model_artifact: Output[Model],
) :
    """Runs the Meridian analysis, saves results to BQ and GCS, and outputs the model."""
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import tensorflow_probability as tfp
    import arviz as az
    import IPython
    from meridian import constants
    from meridian.data import load
    from meridian.data import test_utils
    from meridian.model import model
    from meridian.model import spec
    from meridian.model import prior_distribution
    from meridian.analysis import optimizer
    from meridian.analysis import analyzer
    from meridian.analysis import visualizer
    from meridian.analysis import summarizer
    from meridian.analysis import formatter
    from google.cloud import bigquery
    from google.cloud import storage

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))

    # Load data from the CSV file
    data_df = pd.read_csv(input_data.path + ".csv")



    # --- Meridian Code (Adapted from your notebook) ---
    coord_to_columns = load.CoordToColumns(
        time='time',
        geo='geo',
        controls=['GQV', 'Competitor_Sales'],
        population='population',
        kpi='conversions',
        revenue_per_kpi='revenue_per_conversion',
        media=[
            'Channel0_impression',
            'Channel1_impression',
            'Channel2_impression',
            'Channel3_impression',
            'Channel4_impression',
        ],
        media_spend=[
            'Channel0_spend',
            'Channel1_spend',
            'Channel2_spend',
            'Channel3_spend',
            'Channel4_spend',
        ],
        organic_media=['Organic_channel0_impression'],
        non_media_treatments=['Promo'],
    )

    correct_media_to_channel = {
        'Channel0_impression': 'Channel_0',
        'Channel1_impression': 'Channel_1',
        'Channel2_impression': 'Channel_2',
        'Channel3_impression': 'Channel_3',
        'Channel4_impression': 'Channel_4',
    }
    correct_media_spend_to_channel = {
        'Channel0_spend': 'Channel_0',
        'Channel1_spend': 'Channel_1',
        'Channel2_spend': 'Channel_2',
        'Channel3_spend': 'Channel_3',
        'Channel4_spend': 'Channel_4',
    }

    # Create a DataLoader (mocking CsvDataLoader for simplicity, as we have the df)
    class MockLoader:  # Create a simplified loader
        def __init__(self, df, kpi_type, coord_to_columns, media_to_channel, media_spend_to_channel):
            self.df = df
            self.kpi_type = kpi_type
            self.coord_to_columns = coord_to_columns
            self.media_to_channel = media_to_channel
            self.media_spend_to_channel = media_spend_to_channel

        def load(self):
            return load.load_and_validate_data(
                df_all_geos_all_time=self.df,
                kpi_type=self.kpi_type,
                coord_to_columns=self.coord_to_columns,
                media_to_channel=self.media_to_channel,
                media_spend_to_channel=self.media_spend_to_channel,
			)


    loader = MockLoader(
      df=data_df,
      kpi_type='non_revenue',
      coord_to_columns=coord_to_columns,
      media_to_channel=correct_media_to_channel,
      media_spend_to_channel=correct_media_spend_to_channel,

    )
    data = loader.load()

    roi_mu = 0.2
    roi_sigma = 0.9
    prior = prior_distribution.PriorDistribution(
      roi_m=tfp.distributions.LogNormal(roi_mu, roi_sigma, name=constants.ROI_M)
    )
    model_spec = spec.ModelSpec(prior=prior)
    mmm = model.Meridian(input_data=data, model_spec=model_spec)


    mmm.sample_prior(500)
    mmm.sample_posterior(n_chains=7, n_adapt=500, n_burnin=500, n_keep=1000)

    # Budget Optimization
    budget_optimizer = optimizer.BudgetOptimizer(mmm)
    optimization_results = budget_optimizer.optimize()

    # --- Output to BigQuery (Example: Optimization Results) ---
        # Extract results and convert to DataFrame
    results_df = pd.DataFrame({
         'channel': optimization_results.channels,
         'original_spend': optimization_results.original_spend.numpy(),  # Convert to numpy array
         'original_spend_percent': optimization_results.original_spend_percent.numpy(),
         'original_response': optimization_results.original_response.numpy(),
         'original_roi': optimization_results.original_roi.numpy(),
         'suggested_spend': optimization_results.suggested_spend.numpy(),
         'suggested_spend_percent': optimization_results.suggested_spend_percent.numpy(),
         'suggested_response': optimization_results.suggested_response.numpy(),
         'suggested_roi': optimization_results.suggested_roi.numpy()
    })

    bq_client = bigquery.Client(project=project_id)

    # Write the DataFrame to BigQuery
    job_config = bigquery.LoadJobConfig(
         schema = [
            bigquery.SchemaField("channel", "STRING"),
            bigquery.SchemaField("original_spend", "FLOAT"),
            bigquery.SchemaField("original_spend_percent", "FLOAT"),
            bigquery.SchemaField("original_response", "FLOAT"),
            bigquery.SchemaField("original_roi", "FLOAT"),
            bigquery.SchemaField("suggested_spend", "FLOAT"),
            bigquery.SchemaField("suggested_spend_percent", "FLOAT"),
            bigquery.SchemaField("suggested_response", "FLOAT"),
            bigquery.SchemaField("suggested_roi", "FLOAT"),
        ],

        write_disposition="WRITE_TRUNCATE",  # Overwrite table if it exists
    )

    job = bq_client.load_table_from_dataframe(
        results_df,
        f"{project_id}.{bq_dataset}.{output_bq_table}",
        job_config=job_config
    )
    job.result()  # Wait for the job to complete
    print(f"Optimization results written to BigQuery table: {project_id}.{bq_dataset}.{output_bq_table}")


    # --- Output to GCS (Example: HTML reports and saved models) ---
    mmm_summarizer = summarizer.Summarizer(mmm)
    start_date = '2021-01-25'
    end_date = '2024-01-15'

    # Save Model Summary
    model_summary_filename = "model_summary.html"
    temp_model_summary_path = "/tmp/" + model_summary_filename  # Use a temporary path
    mmm_summarizer.output_model_results_summary(model_summary_filename, "/tmp", start_date, end_date)


    # Save Optimization Summary
    optimization_summary_filename = "optimization_summary.html"
    temp_optimization_summary_path = "/tmp/" + optimization_summary_filename
    optimization_results.output_optimization_summary(optimization_summary_filename, "/tmp")

    # Save the Meridian model
    model_filename = "saved_mmm.pkl"
    temp_model_path = "/tmp/" + model_filename
    model.save_mmm(mmm, temp_model_path)

    # Upload files to GCS
    storage_client = storage.Client(project=project_id)
    bucket_name = gcs_output_dir.split("//")[1].split("/")[0] # Extract bucket name
    bucket_prefix = gcs_output_dir.split(bucket_name + "/")[1]
    bucket = storage_client.bucket(bucket_name)

    def upload_to_gcs(local_path, gcs_path):
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        print(f"File {local_path} uploaded to {gcs_path}.")

    upload_to_gcs(temp_model_summary_path, f"{bucket_prefix}/{model_summary_filename}")
    upload_to_gcs(temp_optimization_summary_path, f"{bucket_prefix}/{optimization_summary_filename}")
    upload_to_gcs(temp_model_path, f"{bucket_prefix}/{model_filename}")

    # --- Save Model Artifact ---
    model_artifact.metadata["framework"] = "Meridian"
    model_artifact.metadata["gcs_path"] = f"{gcs_output_dir}/{model_filename}" # Where the model is
    model.save_mmm(mmm, model_artifact.path + ".pkl") # Serialize the model

    print(f"Meridian analysis complete.  Results saved to GCS: {gcs_output_dir}")

# Define the pipeline
@dsl.pipeline(
    name="meridian-analysis-pipeline",
    description="A pipeline that runs Meridian analysis.",
	pipeline_root=PIPELINE_ROOT,
)
def meridian_pipeline(
    project_id: str = PROJECT_ID,
    bq_dataset: str = BQ_DATASET,
    input_bq_table: str = BQ_TABLE_IN,
    output_bq_table: str = BQ_TABLE_OUT,
    gcs_output_dir: str = GCS_OUTPUT_DIR,
):
    load_data_task = load_data_from_bq(
        project_id=project_id, bq_dataset=bq_dataset, bq_table=input_bq_table
    )

    run_meridian_task = run_meridian_analysis(
        input_data=load_data_task.outputs["output_data"],
        project_id=project_id,
        bq_dataset=bq_dataset,
        output_bq_table=output_bq_table,
        gcs_output_dir=gcs_output_dir,

    )

# Compile the pipeline
compiler.Compiler().compile(
    pipeline_func=meridian_pipeline, package_path="meridian_pipeline.json"
)

# Run the pipeline (using the Vertex AI SDK)
from datetime import datetime

pipeline_job = aiplatform.PipelineJob(
    display_name=DISPLAY_NAME,
    template_path="meridian_pipeline.json",
    job_id=f"meridian-pipeline-{TIMESTAMP}",
    enable_caching=True,  # Consider disabling during development, but enable for production,
    project=PROJECT_ID,
    location=REGION,
)

pipeline_job.run()
print("Pipeline submitted. View in the Vertex AI Pipelines UI.")