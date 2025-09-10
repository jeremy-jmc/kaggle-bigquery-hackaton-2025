import subprocess
import os
import bigframes.pandas as bpd
from google.cloud import bigquery

os.environ['PROJECT_ID'] = 'kaggle-bigquery-471522'

subprocess.run(['gcloud', 'auth', 'login'])
subprocess.run(['gcloud', 'config', 'set', 'project', os.environ['PROJECT_ID']])
subprocess.run(['gcloud', 'auth', 'application-default', 'set-quota-project', os.environ['PROJECT_ID']])

bpd.options.bigquery.project = os.environ['PROJECT_ID']

VALID_INTERACTIONS = "kaggle-bigquery-471522.foodrecsys.valid_interactions_windowed"

df = bpd.read_gbq(VALID_INTERACTIONS)

client = bigquery.Client()


# https://cloud.google.com/python/docs/reference/bigframes/latest
# https://cloud.google.com/bigquery/docs/samples/bigquery-query#bigquery_query-python