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

@dsl.component(
    base_image=CPU_TRAIN_IMAGE,
)
def run_budget_optimization(
    model_artifact: Input[Model],
    output_gcs_dir: str,
    report_filename: str,
    meridian_model_filename: str,
    optimization_report_artifact: Output[Artifact],
):
    # --- This component's *internal* code does not need to change ---
    import os
    import logging
    import time
    import tempfile
    from meridian.analysis import optimizer
    from meridian.model import model
    from google.cloud import storage
    from urllib.parse import urlparse
    import dill # Ensure dill is imported if needed by load_mmm

    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Optional reconfig
    MERIDIAN_MODEL_FILENAME = meridian_model_filename
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
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Model loading failed: {e}")
        raise e

    if not output_gcs_dir.startswith("gs://"):
        raise ValueError("output_gcs_dir must be a GCS path (gs://...)")
    final_gcs_uri = os.path.join(output_gcs_dir, report_filename)

    with tempfile.TemporaryDirectory() as temp_dir:
        logging.info(f"Running optimization and generating report locally in: {temp_dir}")
        local_report_source_path = os.path.join(temp_dir, report_filename)
        try:
            budget_optimizer = optimizer.BudgetOptimizer(mmm)
            optimization_results = budget_optimizer.optimize()
            logging.info("Optimization calculation complete.")
            optimization_results.output_optimization_summary(
                filename=report_filename,
                filepath=temp_dir
            )
            logging.info(f"Meridian saved optimization report locally to: {local_report_source_path}")
            if not os.path.exists(local_report_source_path):
                 logging.error(f"Meridian did not create the expected local report file: {local_report_source_path}")
                 raise FileNotFoundError(f"Optimization report file not created locally by Meridian at {local_report_source_path}")
            logging.info(f"Manually uploading {local_report_source_path} to {final_gcs_uri}")
            upload_local_file_to_gcs(local_report_source_path, final_gcs_uri)
            optimization_report_artifact.uri = final_gcs_uri
            optimization_report_artifact.metadata["gcs_path"] = final_gcs_uri
            optimization_report_artifact.metadata["filename"] = report_filename
            logging.info(f"Set KFP artifact URI to: {optimization_report_artifact.uri}")
        except Exception as e:
            logging.error(f"Failed during budget optimization or reporting/uploading: {e}")
            raise e
    logging.info("Optimization component finished.")
