#!/usr/bin/env sh

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

set -o errexit
set -o nounset

# grep -o '${[A-Z_0-9]\+}' infrastructure/cloudshell/terraform-template.tfvars

. scripts/common.sh
set_environment_variable_from_input_if_not_set "PROJECT_ID" "default project" "Marketing Analytics Jumpstart"
set_environment_variable_if_not_set "MAJ_DEFAULT_PROJECT_ID" "${PROJECT_ID}"
set_environment_variable_from_input_if_not_set "MAJ_DEFAULT_REGION" "default region" "Marketing Analytics Jumpstart"
set_environment_variable_from_input_or_default_if_not_set "MAJ_MDS_PROJECT_ID" "${MAJ_DEFAULT_PROJECT_ID}" "project_id" "Marketing Data Store"
set_environment_variable_from_input_if_not_set "MAJ_MDS_DATA_LOCATION" "location" "BigQuery datasets of Marketing Data Store"
set_environment_variable_from_input_or_default_if_not_set "MAJ_ADS_EXPORT_PROJECT_ID" "${MAJ_GA4_EXPORT_PROJECT_ID}" "project id" "Ads BigQuery export"
export LOCATION=${MAJ_DEFAULT_REGION}
export SOURCE_ROOT=$(pwd)
export TERRAFORM_RUN_DIR=${SOURCE_ROOT}/infrastructure/terraform

set +o nounset
set +o errexit