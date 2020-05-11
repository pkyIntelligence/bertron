from google.cloud import storage
import os


class GCPFileHandler:
    """
    class to control GCP file downloading, using, and deletion
    """
    def __init__(self, bucket_name, source_blob_name, destination_file_name, auth_key_file):
        storage_client = storage.Client.from_service_account_json(auth_key_file)

        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)

        self.filename = destination_file_name

        print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        return

    def __del__(self):
        os.remove(self.filename)