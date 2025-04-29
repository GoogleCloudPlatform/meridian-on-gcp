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

from components.data_analysis import (
    meridian_data_analysis_component
)

CPU_LIMIT = "8"  # vCPUs
MEMORY_LIMIT = "8G"

@dsl.pipeline(
    name="meridian-premodeling-pipeline",
    description="A simple pipeline that imports meridian libs",
)
def data_analysis_pipeline():

    meridian_data_analysis = (
        meridian_data_analysis_component()
        .set_display_name("meridian-data-analysis")
        .set_cpu_limit(CPU_LIMIT)
        .set_memory_limit(MEMORY_LIMIT)
        #.add_node_selector_constraint("NVIDIA_TESLA_T4")
        #.set_gpu_limit(TRAIN_NGPU)
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
