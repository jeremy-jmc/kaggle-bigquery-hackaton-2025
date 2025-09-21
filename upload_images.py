import os
from google.cloud import storage
from tqdm import tqdm
import pandas as pd
import json
import bigframes.pandas as bpd

os.environ['PROJECT_ID'] = 'kaggle-bigquery-471522'
PROJECT_ID = os.environ['PROJECT_ID']

bpd.options.bigquery.project = PROJECT_ID

SCHEMA_NAME = 'foodrecsys'
SUBSET_RECIPE_IDS = f"{PROJECT_ID}.{SCHEMA_NAME}.final_recipes"
RECIPES_ALL = f"{PROJECT_ID}.{SCHEMA_NAME}.recipes"


def upload_folder_to_gcs(bucket_name, source_folder_path, destination_blob_folder, allowed_ids):
    """
    Uploads filtered files from a local folder to a specified folder in a GCS bucket.

    Args:
        bucket_name (str): The name of the GCS bucket.
        source_folder_path (str): The local path of the folder to upload.
        destination_blob_folder (str): The folder path in the GCS bucket.
        allowed_ids (set): A set of recipe IDs (as strings) to be uploaded.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    print(f"Uploading files from '{source_folder_path}' to 'gs://{bucket_name}/{destination_blob_folder}'...")

    print(allowed_ids)
    # Collect all file paths to upload that match the allowed IDs
    all_files = []
    for dirpath, _, filenames in os.walk(source_folder_path):
        for filename in filenames:
            # Extract the file name without extension to get the ID
            file_id = os.path.splitext(filename)[0]
            print(file_id)
            if file_id in allowed_ids:
                all_files.append(os.path.join(dirpath, filename))

    if not all_files:
        print("No matching files found to upload.")
        return

    # Create a tqdm progress bar
    for local_path in tqdm(all_files, desc="Uploading files", unit="file"):
        # Create a relative path to maintain the folder structure
        relative_path = os.path.relpath(local_path, source_folder_path)
        blob_path = os.path.join(destination_blob_folder, relative_path)
        
        blob = bucket.blob(blob_path)
        
        blob.upload_from_filename(local_path)

    print(f"\n{len(all_files)} files have been uploaded to 'gs://{bucket_name}/{destination_blob_folder}'.")


df_recipes = bpd.read_gbq(f"""
SELECT * FROM `{SUBSET_RECIPE_IDS}`
""").to_pandas() # LEFT JOIN `{RECIPES_ALL}` USING(recipe_id)

if __name__ == "__main__":
    BUCKET_NAME = "kaggle-recipes"
    
    SOURCE_FOLDER_PATH = "./data/food_recsys/core-data-images"
    
    DESTINATION_BLOB_FOLDER = "core-data-images"

    if not os.path.isdir(SOURCE_FOLDER_PATH):
        print(f"Error: Source folder '{SOURCE_FOLDER_PATH}' not found.")
    elif BUCKET_NAME == "your-gcs-bucket-name":
        print("Error: Please update the BUCKET_NAME variable with your GCS bucket name.")
    else:
        # Create a set of recipe IDs for efficient lookup.
        # Convert IDs to string to match filenames.
        recipe_ids_to_upload = set(df_recipes['recipe_id'].astype(str))
        
        upload_folder_to_gcs(BUCKET_NAME, SOURCE_FOLDER_PATH, DESTINATION_BLOB_FOLDER, recipe_ids_to_upload)
