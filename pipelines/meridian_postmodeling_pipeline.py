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

from components.modeling import (
    train_meridian_model,
)
from components.post_modeling import (
    run_budget_optimization
)

config_file_path = os.path.join(os.path.dirname(
    __file__), '../config/config.yaml')

base_image = None
gpu_base_image = None
vertex_components_params = None
repo_params = None
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

if vertex_components_params:
    CPU_LIMIT = vertex_components_params['cpu_limit']
    MEMORY_LIMIT = vertex_components_params['memory_limit']
    GPU_LIMIT =  vertex_components_params['gpu_limit']
    ACCELERATOR_TYPE = vertex_components_params['accelerator_type']
else:
    CPU_LIMIT = "8"
    MEMORY_LIMIT = "8G"
    GPU_LIMIT =  1
    ACCELERATOR_TYPE = "NVIDIA_TESLA_T4"

PIPELINE_ROOT = vertex_pipelines_params['root_path']
OUTPUT_GCS_DIR = f"{PIPELINE_ROOT}/outputs"

@dsl.pipeline(
    name="meridian-mmm-gpu-bq-pipeline",
    description="Runs Meridian MMM (GPU) reading from BigQuery, saves summary table to BQ",
    pipeline_root=PIPELINE_ROOT,
)
def post_modeling_pipeline(
    project_id: str,
    location: str,
    mds_dataset: str,
    table_name: str, # Input data table
    roi_mu: float,
    roi_sigma: float,
    n_chains: int,
    n_adapt: int,
    n_burnin: int,
    n_keep: int,
    random_seed: int,
    meridian_model_filename: str = "model_save.pkl",
    optimization_report_filename: str = "optimization_output.html",
):
    # Step 1: Train Model
    train_task = train_meridian_model(
        project_id=project_id,
        bq_dataset=mds_dataset,
        bq_table_name=table_name, # Input table
        roi_mu=roi_mu,
        roi_sigma=roi_sigma,
        n_chains=n_chains,
        n_adapt=n_adapt,
        n_burnin=n_burnin,
        n_keep=n_keep,
        seed=random_seed,
        meridian_model_filename=meridian_model_filename,
    )
    train_task.set_cpu_limit(CPU_LIMIT).set_memory_limit(MEMORY_LIMIT)
    train_task.set_accelerator_limit(GPU_LIMIT).set_accelerator_type(ACCELERATOR_TYPE)

    # Step 4: Run Budget Optimization (Runs in parallel with reports if desired, or after)
    optimization_task = run_budget_optimization(
        model_artifact=train_task.outputs["output_model"],
        output_gcs_dir=OUTPUT_GCS_DIR,
        meridian_model_filename=meridian_model_filename,
        report_filename=optimization_report_filename,
    )
    # Can run after reports by adding: .after(summary_html_task, save_summary_bq_task)
    optimization_task.set_cpu_limit(CPU_LIMIT).set_memory_limit(MEMORY_LIMIT) # Keep original resources
