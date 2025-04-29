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

@component(
    base_image=CPU_TRAIN_IMAGE,
)
def meridian_data_analysis_component() -> str:

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

    from meridian.model.model import Meridian
    from google.cloud import storage
    import joblib
    import os

    # check if GPU is available
    from psutil import virtual_memory
    ram_gb = virtual_memory().total / 1e9
    print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))

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

    loader = load.CsvDataLoader(
        csv_path="https://raw.githubusercontent.com/google/meridian/refs/heads/main/meridian/data/simulated_data/csv/geo_all_channels.csv",
        kpi_type='non_revenue',
        coord_to_columns=coord_to_columns,
        media_to_channel=correct_media_to_channel,
        media_spend_to_channel=correct_media_spend_to_channel,
    )
    data = loader.load()

    return
