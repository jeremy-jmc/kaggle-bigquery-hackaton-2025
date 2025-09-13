import ast
import subprocess
import os
import pandas as pd
import json
import bigframes.pandas as bpd
from google.cloud import bigquery
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

os.environ['PROJECT_ID'] = 'kaggle-bigquery-471522'
PROJECT_ID = os.environ['PROJECT_ID']

subprocess.run(['gcloud', 'auth', 'login'])
subprocess.run(['gcloud', 'config', 'set', 'project', PROJECT_ID])
subprocess.run(['gcloud', 'auth', 'application-default', 'set-quota-project', PROJECT_ID])

bpd.options.bigquery.project = PROJECT_ID

CONNECTION_ID = 'us.kaggle-connection'
SCHEMA_NAME = 'foodrecsys'
VALID_INTERACTIONS = f"{PROJECT_ID}.{SCHEMA_NAME}.valid_interactions_windowed"
TRAIN_INTERACTIONS = f"{PROJECT_ID}.{SCHEMA_NAME}.train_interactions_windowed"
RECIPES_ALL = f"{PROJECT_ID}.{SCHEMA_NAME}.recipes"
SUBSET_RECIPE_IDS = f"{PROJECT_ID}.{SCHEMA_NAME}.final_recipes"

RECIPE_PROFILES_TABLE = f'{SCHEMA_NAME}.recipe_profiles'

df = bpd.read_gbq(VALID_INTERACTIONS)

client = bigquery.Client()

validation_rows = client.query_and_wait(f"""
SELECT * FROM `{VALID_INTERACTIONS}` LIMIT 5
""")
df_validation = validation_rows.to_dataframe()

# -----------------------------------------------------------------------------
# Create a model endpoint in project
# -----------------------------------------------------------------------------
client.query_and_wait(f"""
CREATE OR REPLACE MODEL
  `{SCHEMA_NAME}.gemini_2_5_flash`
REMOTE WITH
    CONNECTION `{CONNECTION_ID}`
    OPTIONS (ENDPOINT = 'gemini-2.5-flash');
""")

client.query_and_wait(f"""
CREATE OR REPLACE MODEL `{SCHEMA_NAME}.text_embedding_model`
REMOTE WITH 
    CONNECTION `{CONNECTION_ID}`
    OPTIONS (ENDPOINT = 'gemini-embedding-001');
""")

# -----------------------------------------------------------------------------
# Generate new columns for `df_recipes` into a new table called `recipe_profiles`
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

df_recipes = bpd.read_gbq(f"""
SELECT * FROM `{SUBSET_RECIPE_IDS}`
LEFT JOIN `{RECIPES_ALL}` USING(recipe_id)
""")

# Convert to pandas DataFrame to use custom functions, then back to BigFrames
df_recipes_pandas = df_recipes.to_pandas()
df_recipes_pandas['parsed_ingredients'] = df_recipes_pandas['ingredients'].apply(prep_ingredients)
df_recipes_pandas['parsed_recipe'] = df_recipes_pandas['cooking_directions'].apply(prep_directions)
df_recipes = bpd.DataFrame(df_recipes_pandas)

# Upload the new table in BigQuery
df_recipes.to_gbq(
    destination_table=f"{PROJECT_ID}.{RECIPE_PROFILES_TABLE}",
    if_exists='replace',
    # project_id=PROJECT_ID
)


# -----------------------------------------------------------------------------
# Recipe Profiles
# -----------------------------------------------------------------------------

class RecipeProfile(BaseModel):
    food_type: str = Field(description="Type of food, e.g., dessert, main course, appetizer")
    cuisine_type: str = Field(description="Cuisine type, e.g., Italian, Chinese, Mexican")
    dietary_preferences: List[str] = Field(description="Dietary preferences, e.g., vegetarian, vegan, gluten-free")
    flavor_profile: List[str] = Field(description="Flavor profile, e.g., spicy, sweet, savory")
    serving_daypart: List[str] = Field(description="Suitable dayparts, e.g., breakfast, lunch, dinner")
    notes: str = Field(description="Short rationale for the profile")


def schema_to_prompt_with_descriptions(model_class) -> str:
    prompt = ""
    for k, v in model_class.model_json_schema()['properties'].items():
        desc = v.get('description', '')
        prompt += f" {k} ({desc}) "
    return f"[ {prompt} ]"


prompt_text = f"Based on the title, following ingredients and cooking directions, create a recipe profile that summarizes the key aspects of the recipe. The answer must follow this structure:{schema_to_prompt_with_descriptions(RecipeProfile)}"

query = f"""
SELECT s.recipe_id, s.title, s.ingredients, s.cooking_directions, s.nutritions, s.reviews, s.parsed_ingredients, s.parsed_recipe, 
AI.GENERATE(('{prompt_text}', s.parsed_ingredients, s.parsed_recipe),
    connection_id => '{CONNECTION_ID}',
    endpoint => 'gemini-2.5-flash',
    model_params => JSON '{{"generationConfig":{{"temperature": 0.5, "maxOutputTokens": 1024, "thinking_config": {{"thinking_budget": 1024}} }} }}',
    output_schema => 'food_type STRING, cuisine_type STRING, dietary_preferences ARRAY<STRING>, flavor_profile ARRAY<STRING>, serving_daypart ARRAY<STRING>, notes STRING'
).full_response AS recipe_profile
FROM (SELECT * FROM `{RECIPE_PROFILES_TABLE}` LIMIT 5) s
"""

print(query)

recipe_rows = client.query_and_wait(query)
df_profiles = recipe_rows.to_dataframe()

response = json.loads(df_profiles['recipe_profile'].iloc[0])
r_profile = json.loads(response['candidates'][0]['content']['parts'][0]['text'])
                      
print(json.dumps(response, indent=2))


# print(df_profiles['combined_text'].iloc[0])


# https://cloud.google.com/python/docs/reference/bigframes/latest
# https://cloud.google.com/bigquery/docs/samples/bigquery-query#bigquery_query-python