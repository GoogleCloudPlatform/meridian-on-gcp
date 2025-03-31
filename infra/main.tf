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

# This file contains the main configuration for Meridian on GCP solution.
# This is the main entry point for the Terraform configuration.
# 
# The configuration is unique for each environment. If you to deploy the solution is a multi-environment scenario,
# you can create a separate Terraform configuration for each environment.
# 
# This solution is designed to be deployed in a Google Cloud project.
# The Terraform backend used is Google Cloud Storage. 
# The Terraform provider used is Google Cloud.
# 
# As a Platform Engineer, you have to keep the terraform.tfvars file and the backend. 
# The terraform.tfvars file contains the configuration values for the solution.
# The backend contains the state file.

# Configure the Google Cloud provider region for this solution. 
# You can set the region in the terraform.tfvars file.
# The default region is us-central1.
# You can deploy and migrate the solution across several regions, check the documentation for more information.

provider "google" {
  region = var.google_default_region
}

data "google_project" "main_project" {
  provider   = google
  project_id = var.main_project_id
}

data "google_project" "data_project" {
  provider   = google
  project_id = var.data_project_id
}

# The locals block contains hardcoded values that are used in the configuration for the solution.
# The locals block is used to define variables that are used in the configuration.
locals {
  # The source_root_dir is the root directory of the project.
  source_root_dir = "."
  # The config_file_name is the name of the config file.
  config_file_name = "config"
  # The uv_run_alias is the alias of the uv run command.
  uv_run_alias = "${var.uv_cmd} run"
  # The project_toml_file_path is the path to the project.toml file.
  project_toml_file_path = "${local.source_root_dir}/pyproject.toml"
  # The project_toml_content_hash is the hash of the project.toml file.
  # This is used for the triggers of the local-exec provisioner.
  project_toml_content_hash = filesha512(local.project_toml_file_path)
  # The generated_sql_queries_directory_path is the path to the generated sql queries directory.
  generated_sql_queries_directory_path = "${local.source_root_dir}/sql/query"
  # The generated_sql_queries_fileset is the list of files in the generated sql queries directory.
  generated_sql_queries_fileset = [for f in fileset(local.generated_sql_queries_directory_path, "*.sqlx") : "${local.generated_sql_queries_directory_path}/${f}"]
  # The generated_sql_queries_content_hash is the sha512 hash of file sha512 hashes in the generated sql queries directory.
  generated_sql_queries_content_hash = sha512(join("", [for f in local.generated_sql_queries_fileset : fileexists(f) ? filesha512(f) : sha512("file-not-found")]))
  # The generated_sql_procedures_directory_path is the path to the generated sql procedures directory.
  generated_sql_procedures_directory_path = "${local.source_root_dir}/sql/procedure"
  # The generated_sql_procedures_fileset is the list of files in the generated sql procedures directory.
  generated_sql_procedures_fileset = [for f in fileset(local.generated_sql_procedures_directory_path, "*.sqlx") : "${local.generated_sql_procedures_directory_path}/${f}"]
  # The generated_sql_procedures_content_hash is the sha512 hash of file sha512 hashes in the generated sql procedures directory.
  generated_sql_procedures_content_hash = sha512(join("", [for f in local.generated_sql_procedures_fileset : fileexists(f) ? filesha512(f) : sha512("file-not-found")]))
}


# Create a configuration file for the solution.
# the template file is located at 
# ${local.source_root_dir}/config/${var.global_config_env}.yaml.tftpl.
# This variable can be set in the terraform.tfvars file. Its default value is "config".
#
#The template file contains the configuration for the feature store. 
#The variables that are replaced with values from the Terraform configuration are:
# project_id: The ID of the Google Cloud project that the feature store will be created in.
# project_name: The name of the Google Cloud project that the feature store will be created in.
# project_number: The number of the Google Cloud project that the feature store will be created in.
# cloud_region: The region in which the feature store will be created.
# mds_project_id: The ID of the Google Cloud project that the feature store will be created in.
resource "local_file" "global_configuration" {
  filename = "${local.source_root_dir}/config/${local.config_file_name}.yaml"
  content = templatefile("${local.source_root_dir}/config/${var.global_config_env}.yaml.tftpl", {
    project_id             = var.main_project_id
    project_name           = data.google_project.main_project.name
    project_number         = data.google_project.main_project.number
    cloud_region           = var.google_default_region
    mds_project_id         = var.data_project_id
    time_zone = var.time_zone
    pipeline_configuration = var.pipeline_configuration
  })
}

# Runs the uv invoke command to generate the sql queries and procedures.
# This command is executed before the feature store is created.
resource "null_resource" "generate_sql_queries" {

  triggers = {
    # The create command generates the sql queries and procedures.
    # The command is: uv inv [function_name] --env-name=${local.config_file_name}
    # The --env-name argument is the name of the configuration file.
    create_command = <<-EOT
    ${local.uv_run_alias} inv apply-config-parameters-to-all-queries --env-name=${local.config_file_name}
    ${local.uv_run_alias} inv apply-config-parameters-to-all-procedures --env-name=${local.config_file_name}
    EOT

    # The destroy command removes the generated sql queries and procedures.
    destroy_command = <<-EOT
    rm -f sql/query/*.sql
    rm -f sql/procedure/*.sql
    EOT

    # The working directory is the root of the project.
    working_dir = local.source_root_dir

    # The source_contents_hash trigger is the hash of the pyproject.toml file.
    # This is used to ensure that the generate_sql_queries command is run only if the pyproject.toml file has changed.
    # It also ensures that the generate_sql_queries command is run only if the sql queries and procedures have changed.
    source_contents_hash        = local_file.global_configuration.content_sha512
    destination_queries_hash    = local.generated_sql_queries_content_hash
    destination_procedures_hash = local.generated_sql_procedures_content_hash
  }

  # Only run the command when `terraform apply` executes and the resource doesn't exist.
  provisioner "local-exec" {
    when        = create
    command     = self.triggers.create_command
    working_dir = self.triggers.working_dir
  }
}