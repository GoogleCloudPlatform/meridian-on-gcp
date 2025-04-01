# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

variable "tf_state_project_id" {
  description = "Google Cloud project where the terraform state file is stored"
  type        = string
}

variable "main_project_id" {
  type        = string
  description = "Project ID where feature store resources are created"
}

variable "data_project_id" {
  description = "Default project to contain the MDS BigQuery datasets"
  type        = string
}

variable "google_default_region" {
  default     = "us-central1"
  description = "The default Google Cloud region."
  type        = string
}

variable "global_config_env" {
  description = "determine which config file is used for globaly for deployment"
  type        = string
  default     = "config"
}

variable "uv_cmd" {
  description = "alias for uv run command on the current system"
  type        = string
  default     = "uv"
}

variable "time_zone" {
  description = "Timezone for scheduled jobs"
  type        = string
  default     = "America/New_York"
}

variable "config_file_path" {
  type        = string
  description = "pipelines config file"
  default     = "config.yaml"
}

variable "uv_run_alias" {
  description = "alias for uv run command on the current system"
  type        = string
  default     = "uv run"
}

variable "pipeline_configuration" {
  description = "Pipeline configuration that will alternate certain settings in the config.yaml.tftpl"
  type = map(
    map(
      object({
        schedule        = object({
          # The `state` defines the state of the pipeline.
          # In case you don't want to schedule the pipeline, set the state to `PAUSED`.
          state                    = string
        })
      })
    )
  )

  default = {
    data-analysis = {
      execution = {
        schedule = {
          state                    = "PAUSED"
        }
      }
    }
    premodeling = {
      execution = {
        schedule = {
          state                    = "PAUSED"
        }
      }
    }
    modeling = {
      execution = {
        schedule = {
          state                    = "PAUSED"
        }
      }
    }
    post-modeling = {
      execution = {
        schedule = {
          state                    = "PAUSED"
        }
      }
    }
  }
  validation {
    condition = alltrue([
      for p in keys(var.pipeline_configuration) : alltrue([
        for c in keys(var.pipeline_configuration[p]) : (
          try(var.pipeline_configuration[p][c].schedule.state, "") == "ACTIVE" ||
          try(var.pipeline_configuration[p][c].schedule.state, "") == "PAUSED"
        )
      ])
    ])
    error_message = "The 'state' field must be either 'PAUSED' or 'ACTIVE' for all pipeline configurations."
  }
}