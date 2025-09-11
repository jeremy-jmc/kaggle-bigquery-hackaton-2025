import ast
import subprocess
import os
import pandas as pd
import bigframes.pandas as bpd
from google.cloud import bigquery

os.environ['PROJECT_ID'] = 'kaggle-bigquery-471522'
PROJECT_ID = os.environ['PROJECT_ID']

subprocess.run(['gcloud', 'auth', 'login'])
subprocess.run(['gcloud', 'config', 'set', 'project', PROJECT_ID])
subprocess.run(['gcloud', 'auth', 'application-default', 'set-quota-project', PROJECT_ID])

bpd.options.bigquery.project = PROJECT_ID

VALID_INTERACTIONS = f"{PROJECT_ID}.foodrecsys.valid_interactions_windowed"
TRAIN_INTERACTIONS = f"{PROJECT_ID}.foodrecsys.train_interactions_windowed"
RECIPES_ALL = f"{PROJECT_ID}.foodrecsys.recipes"
SUBSET_RECIPE_IDS = f"{PROJECT_ID}.foodrecsys.final_recipes"

df = bpd.read_gbq(VALID_INTERACTIONS)

client = bigquery.Client()

validation_rows = client.query_and_wait(f"""
SELECT * FROM `{VALID_INTERACTIONS}` LIMIT 5
""")
df_validation = validation_rows.to_dataframe()


# -----------------------------------------------------------------------------
# Recipe Profiles
# -----------------------------------------------------------------------------


query = f"""
SELECT s.recipe_id, r.title, r.ingredients, r.cooking_directions, r.nutritions, r.reviews, r.parsed_ingredients, r.parsed_recipe, AI.GENERATE(('Based on the title, following ingredients, cooking directions, nutritions, and reviews, create a concise recipe profile that summarizes the key aspects of the recipe. The profile should be engaging and informative, highlighting the main ingredients, cooking method, and any unique features or flavors of the dish. Keep it under 100 words.', r.parsed_ingredients, r.parsed_recipe),
    connection_id => 'us.kaggle-connection',
    endpoint => 'gemini-2.5-flash',
    model_params => JSON '{{"generationConfig":{{"temperature": 0.5, "maxOutputTokens": 1000}}}}'
).result AS recipe_profile
FROM (SELECT recipe_id FROM `{SUBSET_RECIPE_IDS}` LIMIT 5) s
LEFT JOIN `{RECIPES_ALL}` r ON s.recipe_id = r.recipe_id
"""

print(query)

recipe_rows = client.query_and_wait(query)
df_recipes = recipe_rows.to_dataframe()


# -----------------------------------------------------------------------------
# Generate new columns for `df_recipes`
# -----------------------------------------------------------------------------

def prep_ingredients(text: str) -> str:
    if pd.isna(text): return ""
    # Ingredients are caret-separated in your data
    return "\n".join([f"- {v}" for v in str(text).split('^')])


def prep_directions(text: str) -> str:
    if pd.isna(text): return ""
    s = str(text)
    # Some rows look like dict-strings with 'directions' inside; just fall back to raw text
    # Optionally, try to parse if it starts with "{"
    if s.strip().startswith("{"):
        try:
            d = ast.literal_eval(s)
            # common keys: 'directions' (string) or list
            v = d.get('directions', "")
            v = str(v).split('\n')
            v = [x.strip() for x in v if len(x.strip()) > 0]
            v = [f". {x}" if x and x[0].isupper() else x for x in v]

            return " ".join(v).strip(".").replace(" . ", ". ").replace("..", ".").strip()
        except Exception:
            return s.lower()
    return s.lower()

df_recipes['parsed_ingredients'] = df_recipes['ingredients'].apply(prep_ingredients)
df_recipes['parsed_recipe'] = df_recipes['cooking_directions'].apply(prep_directions)

# Upload the new columns to BigQuery
df_recipes_upload = df_recipes[['recipe_id', 'parsed_ingredients', 'parsed_recipe']]
df_recipes_upload.to_gbq(
    destination_table=f"{PROJECT_ID}.foodrecsys.recipes_parsed_temp",
    if_exists='replace',
    project_id=PROJECT_ID
)

# Update the original RECIPES_ALL table with the new columns
client.query_and_wait(f"""
CREATE OR REPLACE TABLE `{RECIPES_ALL}` AS
SELECT 
    r.*,
    p.parsed_ingredients,
    p.parsed_recipe
FROM `{RECIPES_ALL}` r
LEFT JOIN `{PROJECT_ID}.foodrecsys.recipes_parsed_temp` p
ON r.recipe_id = p.recipe_id
""")

# print(df_recipes['combined_text'].iloc[0])


# https://cloud.google.com/python/docs/reference/bigframes/latest
# https://cloud.google.com/bigquery/docs/samples/bigquery-query#bigquery_query-python