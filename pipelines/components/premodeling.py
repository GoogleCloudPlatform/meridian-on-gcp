import json

import google.cloud.aiplatform as aiplatform
from google_cloud_pipeline_components.v1.custom_job import utils
from google_cloud_pipeline_components.preview.custom_job.component import custom_training_job

from kfp import compiler, dsl
import kfp
from kfp.dsl import Artifact, Dataset, Input, Metrics, Model, Output, component

PROJECT_ID = "meridian-dev-455515"
LOCATION = "us-central1"
BUCKET_NAME ="meridian-dev-455515-pipelines"
BUCKET_URI = f"gs://{BUCKET_NAME}"

aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=BUCKET_URI)

TRAIN_GPU, TRAIN_NGPU = (aiplatform.gapic.AcceleratorType.NVIDIA_TESLA_T4, 1)
DEPLOY_GPU, DEPLOY_NGPU = (None, None)

GPU_TRAIN_IMAGE = "us-central1-docker.pkg.dev/meridian-dev-455515/pipelines-docker-repo/meridian-gpu-base-image:dev"
CPU_TRAIN_IMAGE = "us-central1-docker.pkg.dev/meridian-dev-455515/pipelines-docker-repo/meridian-cpu-base-image:dev"

print("GPU Training:", GPU_TRAIN_IMAGE, TRAIN_GPU, TRAIN_NGPU)
print("CPU Training:", CPU_TRAIN_IMAGE, None, None)

MACHINE_TYPE = "n1-standard"

VCPU = "4"
TRAIN_COMPUTE = MACHINE_TYPE + "-" + VCPU
print("Train machine type", TRAIN_COMPUTE)

PIPELINE_ROOT = "{}/pipeline_root/machine_settings".format(BUCKET_URI)

CPU_LIMIT = "8"  # vCPUs
MEMORY_LIMIT = "8G"

@component(
    base_image=CPU_TRAIN_IMAGE,
)
def meridian_model_building_component(
    model_dir: str,
    epochs: int,
) -> str:
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


    def gcs_save_mmm(mmm: Meridian, bucket_name: str, file_path: str):
      storage_client = storage.Client()
      bucket = storage_client.bucket(bucket_name)
      blob = bucket.blob(file_path)

      with blob.open(mode='wb') as f:
        joblib.dump(mmm, f)


    def gcs_load_mmm(bucket_name: str, file_path: str) -> Meridian:
      try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        with blob.open(mode='rb') as f:
          mmm = joblib.load(f)
        return mmm
      except FileNotFoundError:
        raise FileNotFoundError(f"No such file or directory: {file_path}") from None

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

    roi_mu = 0.2     # Mu for ROI prior for each media channel.
    roi_sigma = 0.9  # Sigma for ROI prior for each media channel.
    prior = prior_distribution.PriorDistribution(
        roi_m=tfp.distributions.LogNormal(roi_mu, roi_sigma, name=constants.ROI_M)
    )
    model_spec = spec.ModelSpec(prior=prior)

    mmm = model.Meridian(input_data=data, model_spec=model_spec)

    bucket_name = BUCKET_NAME
    file_path = 'saved_mmm.pkl'

    gcs_save_mmm(mmm, bucket_name, file_path)
    mmm = gcs_load_mmm(bucket_name, file_path)

    return


#compiler.Compiler().compile(meridian_model_building_component, "meridian_model_building_component.yaml")


@component(
    base_image=CPU_TRAIN_IMAGE,
)
def meridian_priors_sampling_component(
    model_dir: str,
    epochs: int,
) -> str:
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


    def gcs_save_mmm(mmm: Meridian, bucket_name: str, file_path: str):
      storage_client = storage.Client()
      bucket = storage_client.bucket(bucket_name)
      blob = bucket.blob(file_path)

      with blob.open(mode='wb') as f:
        joblib.dump(mmm, f)


    def gcs_load_mmm(bucket_name: str, file_path: str) -> Meridian:
      try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        with blob.open(mode='rb') as f:
          mmm = joblib.load(f)
        return mmm
      except FileNotFoundError:
        raise FileNotFoundError(f"No such file or directory: {file_path}") from None

    # check if GPU is available
    from psutil import virtual_memory
    ram_gb = virtual_memory().total / 1e9
    print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))

    bucket_name = BUCKET_NAME
    file_path = 'saved_mmm.pkl'

    mmm = gcs_load_mmm(bucket_name, file_path)

    mmm.sample_prior(500)

    bucket_name = BUCKET_NAME
    file_path = 'saved_sample_prior_mmm.pkl'

    gcs_save_mmm(mmm, bucket_name, file_path)
    mmm = gcs_load_mmm(bucket_name, file_path)

    return


#compiler.Compiler().compile(meridian_priors_sampling_component, "meridian_priors_sampling_component.yaml")


@component(
    base_image=GPU_TRAIN_IMAGE,
)
def meridian_posterior_sampling_component(
    model_dir: str,
    epochs: int,
) -> str:
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


    def gcs_save_mmm(mmm: Meridian, bucket_name: str, file_path: str):
      storage_client = storage.Client()
      bucket = storage_client.bucket(bucket_name)
      blob = bucket.blob(file_path)

      with blob.open(mode='wb') as f:
        joblib.dump(mmm, f)


    def gcs_load_mmm(bucket_name: str, file_path: str) -> Meridian:
      try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        with blob.open(mode='rb') as f:
          mmm = joblib.load(f)
        return mmm
      except FileNotFoundError:
        raise FileNotFoundError(f"No such file or directory: {file_path}") from None

    # check if GPU is available
    from psutil import virtual_memory
    ram_gb = virtual_memory().total / 1e9
    print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))

    bucket_name = BUCKET_NAME
    file_path = 'saved_sample_prior_mmm.pkl'

    mmm = gcs_load_mmm(bucket_name, file_path)

    mmm.sample_posterior(n_chains=7, n_adapt=500, n_burnin=500, n_keep=1000, seed=1)

    bucket_name = BUCKET_NAME
    file_path = 'saved_sample_posterior_mmm.pkl'

    gcs_save_mmm(mmm, bucket_name, file_path)
    mmm = gcs_load_mmm(bucket_name, file_path)

    return


#compiler.Compiler().compile(meridian_posterior_sampling_component, "meridian_posterior_sampling_component.yaml")
