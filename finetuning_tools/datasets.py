from typing import Optional
import os

from finetuning_tools.gcs import BucketStructure


class TuningDatasetMetadata:
    def __init__(self,
                 name: str,
                 bucket_structure: BucketStructure,
                 tfrecord_file_name_all: Optional[str] = None,
                 tfrecord_file_name_train: Optional[str] = None,
                 tfrecord_file_name_val: Optional[str] = None,
                 ):
        self.name = name
        self.bucket_structure = bucket_structure
        self.tfrecord_file_name_all = tfrecord_file_name_all
        self.tfrecord_file_name_train = tfrecord_file_name_train
        self.tfrecord_file_name_val = tfrecord_file_name_val

    @property
    def named_subsets(self):
        subsets = {"all": self.tfrecord_file_name_all,
                   "train": self.tfrecord_file_name_train,
                   "val": self.tfrecord_file_name_val}
        return {k: v for k, v in subsets.items() if v}

    def get_index_path(self, subset_name):
        return os.path.join(self.bucket_structure.indices_dir, f"{self.name}.{subset_name}.index")

    def create_index(self, subset_name):
        tfrecord_file_name = self.named_subsets[subset_name]

        gcs_tfrecord_file_path = os.path.join(self.bucket_structure.gcs_datasets_path, tfrecord_file_name)

        local_index_path = self.get_index_path(subset_name)

        with open(local_index_path, "w") as f:
            f.write(gcs_tfrecord_file_path)

        print(f"wrote index to {local_index_path}")
        return local_index_path

    def upload(self, subset_name=None):
        names = set(self.named_subsets.keys())
        if subset_name:
            names = names.intersection({subset_name})

        print(f"uploading subsets: {names}")

        for name in names:
            filename = self.named_subsets[name]
            self.bucket_structure.upload_dataset(filename)

            local_index_path = self.create_index(subset_name)
            self.bucket_structure.upload_index(local_index_path)
