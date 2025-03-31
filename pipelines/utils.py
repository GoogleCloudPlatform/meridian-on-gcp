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