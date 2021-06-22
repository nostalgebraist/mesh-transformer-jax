import os
from google.cloud import storage


def download_gcs_file(bucket_interface, blob_path, local_directory):
    os.makedirs(local_directory, exist_ok=True)
    blob = bucket_interface.blob(blob_path)

    blob_file_path = blob_path.rpartition('/')[2]
    local_file_path = os.path.join(local_directory, blob_file_path)
    blob.download_to_filename(local_file_path)


def upload_gcs_file(bucket_interface, local_path, gcs_directory):
    local_file_path = local_path.rpartition('/')[2]
    blob_path = os.path.join(gcs_directory, local_file_path)
    blob = bucket_interface.blob(blob_path)

    blob.upload_from_filename(local_path)


class BucketStructure:
    def __init__(self,
                 bucket_name: str,
                 config_dir: str = "configs",
                 model_dir: str = "models",
                 data_dir: str = "data",
                 datasets_subdir: str = "datasets",
                 indices_subdir: str = "indices",
                 ):
        self.bucket_name = bucket_name
        self.config_dir = config_dir
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.datasets_dir = os.path.join(self.data_dir, datasets_subdir)
        self.indices_dir = os.path.join(self.data_dir, indices_subdir)

        self.client = storage.Client()
        self.bucket_interface = self.client.bucket(self.bucket_name)

    @property
    def gcs_config_path(self):
        return os.path.join("gs://", self.bucket_name, self.config_dir)

    @property
    def gcs_model_path(self):
        return os.path.join("gs://", self.bucket_name, self.model_dir)

    @property
    def gcs_datasets_path(self):
        return os.path.join("gs://", self.bucket_name, self.datasets_dir)

    @property
    def gcs_indices_path(self):
        return os.path.join("gs://", self.bucket_name, self.indices_dir)

    def upload_config(self, name):
        local_path = os.path.join(self.config_dir, name)
        upload_gcs_file(self.bucket_interface, local_path=local_path, gcs_directory=self.gcs_config_path)

    def upload_dataset(self, name):
        local_path = os.path.join(self.gcs_datasets_path, name)
        upload_gcs_file(self.bucket_interface, local_path=local_path, gcs_directory=self.gcs_datasets_path)

    def upload_index(self, name):
        local_path = os.path.join(self.gcs_indices_path, name)
        upload_gcs_file(self.bucket_interface, local_path=local_path, gcs_directory=self.gcs_indices_path)

    def download_config(self, name):
        blob_path = os.path.join(self.gcs_config_path, name)
        download_gcs_file(self.bucket_interface, blob_path=blob_path, local_directory=self.config_dir)

    def download_dataset(self, name):
        blob_path = os.path.join(self.gcs_datasets_path, name)
        download_gcs_file(self.bucket_interface, blob_path=blob_path, local_directory=self.datasets_dir)

    def download_index(self, name):
        blob_path = os.path.join(self.gcs_indices_path, name)
        download_gcs_file(self.bucket_interface, blob_path=blob_path, local_directory=self.indices_dir)
