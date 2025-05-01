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

from typing import Optional
import kfp as kfp
import kfp.dsl as dsl
import logging # Import logging
from typing import Optional

from components.data_analysis import (
    meridian_data_analysis_component
)

import os
import yaml

config_file_path = os.path.join(os.path.dirname(
    __file__), '../config/config.yaml')

vertex_components_params = None
if os.path.exists(config_file_path):
    with open(config_file_path, encoding='utf-8') as fh:
        configs = yaml.full_load(fh)

    vertex_components_params = configs['vertex_ai']['components']

if vertex_components_params:
    CPU_LIMIT = vertex_components_params['cpu_limit']
    MEMORY_LIMIT = vertex_components_params['memory_limit']
else:
    CPU_LIMIT = "8"
    MEMORY_LIMIT = "8G"


@dsl.pipeline(
    name="meridian-premodeling-pipeline",
    description="A simple pipeline that performs data analysis",
)
def data_analysis_pipeline(
    project_id: str,
    location: str,
    mds_dataset: str,
    table_name: str, # Input data table
):

    meridian_data_analysis = (
        meridian_data_analysis_component(
            project_id=project_id,
            bq_dataset=mds_dataset,
            bq_table_name=table_name,
        )
        .set_display_name("meridian-data-analysis")
        .set_cpu_limit(CPU_LIMIT)
        .set_memory_limit(MEMORY_LIMIT)
    )

    #meridian_model_building = (
    #    meridian_model_building_component(epochs=epochs, model_dir=model_dir)
    #    .set_display_name("meridian-model-building")
    #    .set_cpu_limit(CPU_LIMIT)
    #    .set_memory_limit(MEMORY_LIMIT)
        #.add_node_selector_constraint("NVIDIA_TESLA_T4")
        #.set_gpu_limit(TRAIN_NGPU)
    #)
    #meridian_model_building.after(meridian_data_analysis)


    #meridian_sample_prior = (
    #    meridian_priors_sampling_component(epochs=epochs, model_dir=model_dir)
    #    .set_display_name("meridian-sample-prior")
    #    .set_cpu_limit(CPU_LIMIT)
    #    .set_memory_limit(MEMORY_LIMIT)
        #.add_node_selector_constraint("NVIDIA_TESLA_T4")
        #.set_gpu_limit(TRAIN_NGPU)
    #)
    #meridian_sample_prior.after(meridian_model_building)

    #meridian_sample_posterior = (
    #    meridian_posterior_sampling_component(epochs=epochs, model_dir=model_dir)
    #    .set_display_name("meridian-sample-posterior")
    #    .set_cpu_limit(CPU_LIMIT)
    #    .set_memory_limit(MEMORY_LIMIT)
    #    .add_node_selector_constraint("NVIDIA_TESLA_T4")
    #    .set_gpu_limit(TRAIN_NGPU)
    #)
    #meridian_sample_posterior.after(meridian_sample_prior)
