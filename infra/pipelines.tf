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

data "local_file" "config_vars" {
  filename = var.config_file_path
}

# The locals block is used to define local variables that can be used within the Terraform module.
locals {
  # This variable stores the parsed contents of the YAML configuration file.
  config_vars                              = yamldecode(data.local_file.config_vars.content)
  cloud_build_vars                         = local.config_vars.cloud_build
  artifact_registry_vars                   = local.config_vars.artifact_registry
  pipeline_image_vars                      = local.config_vars.vertex_ai.components
  pipeline_vars                            = local.config_vars.vertex_ai.pipelines
  dataflow_vars                            = local.config_vars.dataflow
  config_bigquery                          = local.config_vars.bigquery
  source_root_dir                          = "../.."
  config_file_path_relative_python_run_dir = substr(var.config_file_path, 3, length(var.config_file_path))
  compile_pipelines_tag                    = "v1"
}


# This resource creates a service account to run the Vertex AI pipelines
resource "google_service_account" "service_account" {
  project      = local.pipeline_vars.project_id
  account_id   = local.pipeline_vars.service_account_id
  display_name = local.pipeline_vars.service_account_id
  description  = "Service Account to run Vertex AI Pipelines"

  # The lifecycle block is used to configure the lifecycle of the table. In this case, the ignore_changes attribute is set to all, which means that Terraform will ignore 
  # any changes to the table and will not attempt to update the table. The prevent_destroy attribute is set to true, which means that Terraform will prevent the table from being destroyed.
  lifecycle {
    ignore_changes  = all
    #prevent_destroy = true
    create_before_destroy = true
  }
}

# Wait for the pipelines service account to be created
resource "null_resource" "wait_for_vertex_pipelines_sa_creation" {
  provisioner "local-exec" {
    command = <<-EOT
    COUNTER=0
    MAX_TRIES=100
    while ! gcloud iam service-accounts list --project=${local.pipeline_vars.project_id} --filter="EMAIL:${local.pipeline_vars.service_account} AND DISABLED:False" --format="table(EMAIL, DISABLED)" && [ $COUNTER -lt $MAX_TRIES ]
    do
      sleep 3
      printf "."
      COUNTER=$((COUNTER + 1))
    done
    if [ $COUNTER -eq $MAX_TRIES ]; then
      echo "pipelines service account was not created, terraform can not continue!"
      exit 1
    fi
    sleep 20
    EOT
  }

  depends_on = [
    google_service_account.service_account
  ]
}


# This resource binds the service account to the required roles
resource "google_project_iam_member" "pipelines_sa_roles" {
  depends_on = [
    null_resource.wait_for_vertex_pipelines_sa_creation
  ]

  project = local.pipeline_vars.project_id
  member  = "serviceAccount:${google_service_account.service_account.email}"

  for_each = toset([
    "roles/iap.tunnelResourceAccessor",
    "roles/compute.osLogin",
    "roles/bigquery.jobUser",
    "roles/bigquery.dataEditor",
    "roles/storage.admin",
    "roles/aiplatform.user",
    "roles/artifactregistry.reader",
    "roles/pubsub.publisher",
    "roles/dataflow.developer",
    "roles/bigquery.connectionUser",
    "roles/compute.networkUser"
  ])
  role = each.key

  # The lifecycle block is used to configure the lifecycle of the table. In this case, the ignore_changes attribute is set to all, which means that Terraform will ignore 
  # any changes to the table and will not attempt to update the table. The prevent_destroy attribute is set to true, which means that Terraform will prevent the table from being destroyed.
  lifecycle {
    ignore_changes  = all
    #prevent_destroy = true
    create_before_destroy = true
  }
}

# This resource binds the service account to the required roles in the mds project
resource "google_project_iam_member" "pipelines_sa_mds_project_roles" {
  depends_on = [
    null_resource.wait_for_vertex_pipelines_sa_creation
  ]

  project = local.pipeline_vars.project_id
  member  = "serviceAccount:${google_service_account.service_account.email}"

  for_each = toset([
    "roles/bigquery.dataViewer"
  ])
  role = each.key

  # The lifecycle block is used to configure the lifecycle of the table. In this case, the ignore_changes attribute is set to all, which means that Terraform will ignore 
  # any changes to the table and will not attempt to update the table. The prevent_destroy attribute is set to true, which means that Terraform will prevent the table from being destroyed.
  lifecycle {
    ignore_changes  = all
    #prevent_destroy = true
    create_before_destroy = true
  }
}

# This resource creates a service account to run the dataflow jobs
resource "google_service_account" "dataflow_worker_service_account" {
  project      = local.pipeline_vars.project_id
  account_id   = local.dataflow_vars.worker_service_account_id
  display_name = local.dataflow_vars.worker_service_account_id
  description  = "Service Account to run Dataflow jobs"

  # The lifecycle block is used to configure the lifecycle of the table. In this case, the ignore_changes attribute is set to all, which means that Terraform will ignore 
  # any changes to the table and will not attempt to update the table. The prevent_destroy attribute is set to true, which means that Terraform will prevent the table from being destroyed.
  lifecycle {
    ignore_changes  = all
    #prevent_destroy = true
    create_before_destroy = true
  }
}

# Wait for the dataflow worker service account to be created
resource "null_resource" "wait_for_dataflow_worker_sa_creation" {
  provisioner "local-exec" {
    command = <<-EOT
    COUNTER=0
    MAX_TRIES=100
    while ! gcloud iam service-accounts list --project=${local.pipeline_vars.project_id} --filter="EMAIL:${local.dataflow_vars.worker_service_account} AND DISABLED:False" --format="table(EMAIL, DISABLED)" && [ $COUNTER -lt $MAX_TRIES ]
    do
      sleep 3
      printf "."
      COUNTER=$((COUNTER + 1))
    done
    if [ $COUNTER -eq $MAX_TRIES ]; then
      echo "dataflow worker service account was not created, terraform can not continue!"
      exit 1
    fi
    sleep 20
    EOT
  }

  depends_on = [
    google_service_account.dataflow_worker_service_account
  ]
}

# This resource binds the service account to the required roles
resource "google_project_iam_member" "dataflow_worker_sa_roles" {
  depends_on = [
    null_resource.wait_for_dataflow_worker_sa_creation
  ]

  project = local.pipeline_vars.project_id
  member  = "serviceAccount:${google_service_account.dataflow_worker_service_account.email}"

  for_each = toset([
    "roles/dataflow.worker",
    "roles/bigquery.dataEditor",
    "roles/bigquery.jobUser",
    "roles/storage.objectAdmin",
  ])
  role = each.key

  # The lifecycle block is used to configure the lifecycle of the table. In this case, the ignore_changes attribute is set to all, which means that Terraform will ignore 
  # any changes to the table and will not attempt to update the table. The prevent_destroy attribute is set to true, which means that Terraform will prevent the table from being destroyed.
  lifecycle {
    ignore_changes  = all
    #prevent_destroy = true
    create_before_destroy = true
  }
}

# This resource binds the service account to the required roles
# Allow pipelines SA service account use dataflow worker SA
resource "google_service_account_iam_member" "dataflow_sa_iam" {
  depends_on = [
    null_resource.wait_for_dataflow_worker_sa_creation
  ]

  service_account_id = "projects/${local.pipeline_vars.project_id}/serviceAccounts/${google_service_account.dataflow_worker_service_account.email}"
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.service_account.email}"

  # The lifecycle block is used to configure the lifecycle of the table. In this case, the ignore_changes attribute is set to all, which means that Terraform will ignore 
  # any changes to the table and will not attempt to update the table. The prevent_destroy attribute is set to true, which means that Terraform will prevent the table from being destroyed.
  lifecycle {
    ignore_changes  = all
    #prevent_destroy = true
    create_before_destroy = true
  }
}

# This resource creates a Cloud Storage Bucket for the pipeline artifacts
resource "google_storage_bucket" "pipelines_bucket" {
  project                     = local.pipeline_vars.project_id
  name                        = local.pipeline_vars.bucket_name
  storage_class               = "REGIONAL"
  location                    = local.pipeline_vars.region
  uniform_bucket_level_access = true
  # The force_destroy attribute specifies whether the bucket should be forcibly destroyed 
  # even if it contains objects. In this case, it's set to false, which means that the bucket will not be destroyed if it contains objects.
  force_destroy = false

  # The lifecycle block is used to configure the lifecycle of the table. In this case, the ignore_changes attribute is set to all, which means that Terraform will ignore 
  # any changes to the table and will not attempt to update the table. The prevent_destroy attribute is set to true, which means that Terraform will prevent the table from being destroyed.
  lifecycle {
    ignore_changes  = all
    #prevent_destroy = true
    create_before_destroy = true
  }
}

# This resource creates a Cloud Storage Bucket for the model assets
resource "google_storage_bucket" "custom_model_bucket" {
  project                     = local.pipeline_vars.project_id
  name                        = local.pipeline_vars.model_bucket_name
  storage_class               = "REGIONAL"
  location                    = local.pipeline_vars.region
  uniform_bucket_level_access = true
  # The force_destroy attribute specifies whether the bucket should be forcibly destroyed 
  # even if it contains objects. In this case, it's set to false, which means that the bucket will not be destroyed if it contains objects.
  force_destroy = false

  # The lifecycle block is used to configure the lifecycle of the table. In this case, the ignore_changes attribute is set to all, which means that Terraform will ignore 
  # any changes to the table and will not attempt to update the table. The prevent_destroy attribute is set to true, which means that Terraform will prevent the table from being destroyed.
  lifecycle {
    ignore_changes  = all
    #prevent_destroy = true
    create_before_destroy = true
  }
}

# The locals block defines a local variable named vertex_pipelines_available_locations that contains a list of 
# all the available regions for Vertex AI Pipelines. 
# This variable is used to validate the value of the location attribute of the google_artifact_registry_repository resource.
locals {
  vertex_pipelines_available_locations = [
    "asia-east1",
    "asia-east2",
    "asia-northeast1",
    "asia-northeast2",
    "asia-northeast3",
    "asia-south1",
    "asia-southeast1",
    "asia-southeast2",
    "europe-central2",
    "europe-north1",
    "europe-west1",
    "europe-west2",
    "europe-west3",
    "europe-west4",
    "europe-west6",
    "europe-west8",
    "europe-west9",
    "europe-southwest1",
    "me-west1",
    "northamerica-northeast1",
    "northamerica-northeast2",
    "southamerica-east1",
    "southamerica-west1",
    "us-central1",
    "us-east1",
    "us-east4",
    "us-south1",
    "us-west1",
    "us-west2",
    "us-west3",
    "us-west4",
    "australia-southeast1",
    "australia-southeast2",
  ]
}

# This resource creates an Artifact Registry repository for the pipeline artifacts
resource "google_artifact_registry_repository" "pipelines-repo" {
  project       = local.pipeline_vars.project_id
  location      = local.artifact_registry_vars.pipelines_repo.region
  repository_id = local.artifact_registry_vars.pipelines_repo.name
  description   = "Pipelines Repository"
  # The format is kubeflow pipelines YAML files.
  format = "KFP"

  # The lifecycle block of the google_artifact_registry_repository resource defines a precondition that 
  # checks if the specified region is included in the vertex_pipelines_available_locations list. 
  # If the condition is not met, an error message is displayed and the Terraform configuration will fail.
  lifecycle {
    precondition {
      condition     = contains(local.vertex_pipelines_available_locations, local.artifact_registry_vars.pipelines_repo.region)
      error_message = "Vertex AI Pipelines is not available in your default region: ${local.artifact_registry_vars.pipelines_repo.region}.\nSet 'google_default_region' variable to a valid Vertex AI Pipelines location, see https://cloud.google.com/vertex-ai/docs/general/locations."
    }
  }
}

# This resource creates an Artifact Registry repository for the pipeline docker images
resource "google_artifact_registry_repository" "pipelines_docker_repo" {
  project       = local.pipeline_vars.project_id
  location      = local.artifact_registry_vars.pipelines_docker_repo.region
  repository_id = local.artifact_registry_vars.pipelines_docker_repo.name
  description   = "Docker Images Repository"
  # The format is Docker images.
  format = "DOCKER"
}

locals {
  base_component_image_dir = "${local.source_root_dir}/python/base_component_image"
  component_image_fileset = [
    #"${local.base_component_image_dir}/build-push.py",
    "${local.base_component_image_dir}/BaseImage.dockerfile",
    "${local.base_component_image_dir}/GPUBaseImage.dockerfile",
    #"${local.base_component_image_dir}/pyproject.toml",
    #"${local.base_component_image_dir}/components/vertex.py",
  ]
  # This is the content of the hash of all the files related to the base component image used to run each
  # Vertex AI Pipeline step.
  component_image_content_hash = sha512(join("", [for f in local.component_image_fileset : fileexists(f) ? filesha512(f) : sha512("file-not-found")]))

  pipelines_dir = "${local.source_root_dir}/python/pipelines"
  pipelines_fileset = [
    #"${local.pipelines_dir}/components/bigquery/component.py",
    #"${local.pipelines_dir}/components/pubsub/component.py",
    #"${local.pipelines_dir}/components/vertex/component.py",
    #"${local.pipelines_dir}/components/python/component.py",
    "${local.pipelines_dir}/compiler.py",
    #"${local.pipelines_dir}/feature_engineering_pipelines.py",
    "${local.pipelines_dir}/pipeline_ops.py",
    "${local.pipelines_dir}/scheduler.py",
    #"${local.pipelines_dir}/segmentation_pipelines.py",
    #"${local.pipelines_dir}/auto_segmentation_pipelines.py",
    #"${local.pipelines_dir}/tabular_pipelines.py",
    "${local.pipelines_dir}/uploader.py",
  ]
  # This is the content of the hash of all the files related to the pipelines definitions used to run each
  # Vertex AI Pipeline.
  pipelines_content_hash = sha512(join("", [for f in local.pipelines_fileset : fileexists(f) ? filesha512(f) : sha512("file-not-found")]))
}

# This resource is used to build and push the base component image that will be used to run each Vertex AI Pipeline step.
resource "null_resource" "build_push_pipelines_components_image" {
  triggers = {
    working_dir             = "${local.source_root_dir}/components"
    docker_repo_id          = google_artifact_registry_repository.pipelines_docker_repo.id
    docker_repo_create_time = google_artifact_registry_repository.pipelines_docker_repo.create_time
    source_content_hash     = local.component_image_content_hash
  }

  # The provisioner block specifies the command that will be executed to build and push the base component image.
  # This command will execute the build-push function in the base_component_image module, which will build and push the base component image to the specified Docker repository.
  provisioner "local-exec" {
    command     = "${var.uv_run_alias} python -m image_builder -c ${local.config_file_path_relative_python_run_dir}"
    working_dir = self.triggers.working_dir
  }
}


# Wait for the dataflow worker service account to be created
resource "null_resource" "check_pipeline_docker_image_pushed" {
  provisioner "local-exec" {
    command = <<-EOT
    COUNTER=0
    MAX_TRIES=100
    while ! gcloud artifacts docker images list --project=${local.pipeline_vars.project_id} ${local.artifact_registry_vars.pipelines_docker_repo.region}-docker.pkg.dev/${local.pipeline_vars.project_id}/${local.artifact_registry_vars.pipelines_docker_repo.name} --format="table(IMAGE, CREATE_TIME, UPDATE_TIME)" && [ $COUNTER -lt $MAX_TRIES ]
    do
      sleep 5
      printf "."
      COUNTER=$((COUNTER + 1))
    done
    if [ $COUNTER -eq $MAX_TRIES ]; then
      echo "pipeline docker image was not created, terraform can not continue!"
      exit 1
    fi
    sleep 20
    EOT
  }

  depends_on = [
    null_resource.build_push_pipelines_components_image
  ]
}