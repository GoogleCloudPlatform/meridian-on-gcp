# Installation Guide

## Step 1 - Select a Google Cloud project and open Cloud Shell
In the Google Cloud Console, navigate to your project and open Cloud Shell. 

Note: Make sure you have Owner privileges.


## Step 2 - Initial environment setup
1. Run the following commands to setup environment variables.

```bash
export PROPJECT_ID=YOUR_PROJECT_ID
gcloud config set project $PROJECT_ID
cd ${HOME}
REPO="meridian-on-gcp"
git clone https://github.com/GoogleCloudPlatform/meridian-on-gcp.git
```

1. Install update uv for running python scripts Install uv that manages the python version and dependecies for the solution.
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

Check uv installation:
```bash
uv --version
```

1. Run the following script to configure the local environment.

```bash
SOURCE_ROOT="${HOME}/${REPO}"
cd ${SOURCE_ROOT}
scripts/set-env.sh
```

1. Authenticate with additional OAuth 2.0 scopes needed to use the Google Analytics Admin API:
```bash
gcloud auth login
gcloud auth application-default login --quiet --scopes="openid,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/accounts.reauth"
gcloud auth application-default set-quota-project $PROJECT_ID
CREDENTIAL_FILE=`gcloud auth application-default set-quota-project "${PROJECT_ID}" 2>&1 | grep -e "Credentials saved to file:" | cut -d "[" -f2 | cut -d "]" -f1`
export GOOGLE_APPLICATION_CREDENTIALS=$CREDENTIAL_FILE
```

Note: You may receive an error message informing the Cloud Resource Manager API has not been used/enabled for your project, similar to the following:

ERROR: (gcloud.auth.application-default.login) User [@.com] does not have permission to access projects instance [<gcp_project_ID>:testIamPermissions] (or it may not exist): Cloud Resource Manager API has not been used in project <gcp_project_id> before or it is disabled. Enable it by visiting https://console.developers.google.com/apis/api/cloudresourcemanager.googleapis.com/overview?project=<gcp_project_id> then retry. If you enabled this API recently, wait a few minutes for the action to propagate to our systems and retry.

On the next step, the Cloud Resource Manager API will be enabled and, then, your credentials will finally work.

1. Review your Terraform version

Make sure you have installed terraform version is 1.9.7. We recommend you to use tfenv to manage your terraform version. Tfenv is a version manager inspired by rbenv, a Ruby programming language version manager.

To install tfenv, run the following commands:
```bash
# Install via Homebrew or via Arch User Repository (AUR)
# Follow instructions on https://github.com/tfutils/tfenv

# Now, install the recommended terraform version 
tfenv install 1.9.7
tfenv use 1.9.7
terraform --version
```
Note: If you have a Apple Silicon Macbook, you should install terraform by setting the TFENV_ARCH environment variable:
```bash
TFENV_ARCH=amd64 tfenv install 1.9.7
tfenv use 1.9.7
terraform --version
```
If not properly terraform version for your architecture is installed, terraform .. init will fail.

For instance, the output on MacOS should be like:
```bash
Terraform v1.9.7
on darwin_amd64
```

## Step 3 - Define Terraform Variables file
Create the Terraform variables file by making a copy from the template and set the Terraform variables. Most of the parameters are based on the pre-requisites described here. The sample file has all the required variables listed.

```bash
SOURCE_ROOT=${HOME}/${REPO}
TERRAFORM_RUN_DIR=${SOURCE_ROOT}/infra
cd ${TERRAFORM_RUN_DIR}
cp ${TERRAFORM_RUN_DIR}/terraform-sample.tfvars ${TERRAFORM_RUN_DIR}/terraform.tfvars
```

Edit the variables file. If using Vim:
```bash
vim ${TERRAFORM_RUN_DIR}/terraform.tfvars
```
Note: The variable google_default_region determines the region where the resources are hosted. The variable default value is us-central1, based on your data residency requirements you should change the variable value by add the following in your terraform.tfvars file:
```bash
google_default_region = "[specific Google Cloud region of choice]"
```
Note: The variable destination_data_location determines the location for the data store in BigQuery. You have the choice to either store the data in single region by assigning value such as us-central1, europe-west1, asia-east1 etc or in multi-regions by assigning value such as US or EU.

## Step 4 - Apply Terraform configuration
This step may take a few hours so be patient. Reason: the steps to build two CUDA images have been moved as part of the terraform script (pipeline_images.tf). You'll need to monitor the 1.5-2 hours long image build job, either from Log Explorer or from the URL generated from the terraform apply output. You may leave the Cloud Shell as it is a background process. This is an improved process from previous version where you'll need to keep the terminal session running to complete the image build process.

```bash
terraform -chdir="${TERRAFORM_RUN_DIR}" init
terraform -chdir="${TERRAFORM_RUN_DIR}" plan
terraform -chdir="${TERRAFORM_RUN_DIR}" validate
terraform -chdir="${TERRAFORM_RUN_DIR}" apply
```

If you don't have a successful execution of certain resources, re-run `terraform -chdir="${TERRAFORM_RUN_DIR}" apply` a few more times until all is deployed successfully. However, if there are still resources not deployed, open a new github issue.