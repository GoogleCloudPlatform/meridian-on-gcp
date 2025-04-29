#Copyright 2025 Google LLC
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
from kfp import dsl
from kfp.dsl import Input, Output, Model, Dataset, Artifact, OutputPath
from kfp.dsl import Artifact, Dataset, Input, Metrics, Model, Output, component
from typing import NamedTuple, Optional
import google.cloud.aiplatform as aip
import os
import shutil
import tempfile
import logging # Import logging
from typing import Optional
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


# --- Configuration ---
PROJECT_ID = "YOUR-PROJECT-ID"
REGION = "us-central1"
PIPELINE_ROOT = "{}/pipeline_root/machine_settings".format(BUCKET_URI)
BQ_DATASET = "meridiansampledataset" # Your dataset
BQ_TABLE_NAME = "meridiantable" # your table name
BQ_SUMMARY_TABLE_NAME = "meridian_media_summary_report" # Name for the new BQ out table
OUTPUT_GCS_DIR = f"{PIPELINE_ROOT}/outputs"
ROI_MU = 0.2
ROI_SIGMA = 0.9
N_CHAINS = 7
N_ADAPT = 500
N_BURNIN = 500
N_KEEP = 1000
RANDOM_SEED = 1
REPORT_START_DATE = '2021-01-25'
REPORT_END_DATE = '2024-01-15'
STANDARD_BASE_IMAGE = "python:3.10-slim"
GPU_BASE_IMAGE = "gcr.io/deeplearning-platform-release/tf-gpu.2-15.py310"
MERIDIAN_MODEL_FILENAME = "model_save.pkl"
PIPELINE_NAME = "meridian-mmm-gpu-bq-pipeline" # pipeline name
PIPELINE_JSON = f"{PIPELINE_NAME}.json"

# --- Configure logging for components ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
@dsl.pipeline(
    name=PIPELINE_NAME,
    description="Runs Meridian MMM (GPU) reading from BigQuery, saves summary table to BQ",
    pipeline_root=PIPELINE_ROOT,
)
def meridian_pipeline(
    project_id: str = PROJECT_ID,
    bq_dataset: str = BQ_DATASET,
    bq_table_name: str = BQ_TABLE_NAME, # Input data table
    summary_bq_table_name: str = BQ_SUMMARY_TABLE_NAME, # Output summary table
    output_gcs_dir: str = OUTPUT_GCS_DIR,
    roi_mu: float = ROI_MU,
    roi_sigma: float = ROI_SIGMA,
    n_chains: int = N_CHAINS,
    n_adapt: int = N_ADAPT,
    n_burnin: int = N_BURNIN,
    n_keep: int = N_KEEP,
    seed: int = RANDOM_SEED,
    report_start_date: str = REPORT_START_DATE,
    report_end_date: str = REPORT_END_DATE,
    summary_report_filename: str = "summary_output.html", # HTML report
    optimization_report_filename: str = "optimization_output.html",
):
    # Step 1: Train Model
    train_task = train_meridian_model(
        project_id=project_id,
        bq_dataset=bq_dataset,
        bq_table_name=bq_table_name, # Input table
        roi_mu=roi_mu, roi_sigma=roi_sigma,
        n_chains=n_chains, n_adapt=n_adapt, n_burnin=n_burnin, n_keep=n_keep, seed=seed,
    )
    train_task.set_cpu_limit("16").set_memory_limit("64G")
    train_task.set_accelerator_limit(1).set_accelerator_type('NVIDIA_TESLA_T4')

    # Step 2: Generate Summary Table and Save to BigQuery
    save_summary_bq_task = generate_and_save_summary_bq(
        model_artifact=train_task.outputs["output_model"],
        project_id=project_id,
        bq_dataset=bq_dataset,
        bq_table_name=summary_bq_table_name, # Output table name for summary
    )
    save_summary_bq_task.set_cpu_limit("16").set_memory_limit("64G") # Adjust resources as needed

    # Step 3: Generate HTML Summary Report (Runs in parallel with BQ save if desired, or after)
    summary_html_task = generate_summary_report(
        model_artifact=train_task.outputs["output_model"],
        output_gcs_dir=output_gcs_dir,
        report_filename=summary_report_filename,
        start_date=report_start_date,
        end_date=report_end_date,
    )
    # Can run after BQ save by adding: .after(save_summary_bq_task)
    summary_html_task.set_cpu_limit("16").set_memory_limit("64G") # Keep original resources

    # Step 4: Run Budget Optimization (Runs in parallel with reports if desired, or after)
    optimization_task = run_budget_optimization(
        model_artifact=train_task.outputs["output_model"],
        output_gcs_dir=output_gcs_dir,
        report_filename=optimization_report_filename,
    )
    # Can run after reports by adding: .after(summary_html_task, save_summary_bq_task)
    optimization_task.set_cpu_limit("16").set_memory_limit("64G") # Keep original resources
