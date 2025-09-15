import ast
import subprocess
import os
import pandas as pd
import json
import bigframes.pandas as bpd
from google.cloud import bigquery
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from tqdm import tqdm
from dotenv import load_dotenv
from IPython.display import display
load_dotenv()
pd.set_option('display.max_colwidth', 100)

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

RECIPES_PARSED = f'{SCHEMA_NAME}.recipes_parsed'
RECIPES_PROFILES_TABLE = f"{SCHEMA_NAME}.recipe_profiles"

USERS_PARSED = f'{SCHEMA_NAME}.users_parsed'
USERS_PROFILES_TABLE = f"{SCHEMA_NAME}.user_profiles"

VECTOR_SEARCH_RESULTS_TABLE = f"{SCHEMA_NAME}.vector_search_results"

df = bpd.read_gbq(VALID_INTERACTIONS)

client = bigquery.Client()

def schema_to_prompt_with_descriptions(model_class) -> str:
    prompt = ""
    for k, v in model_class.model_json_schema()['properties'].items():
        desc = v.get('description', '')
        prompt += f" {k} ({desc}) "
    return f"[ {prompt} ]"


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

# * RECIPES
# -----------------------------------------------------------------------------
# PARSING: Generate new parsed columns for `df_recipes` into a new table called `recipe_profiles`
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
    destination_table=f"{PROJECT_ID}.{RECIPES_PARSED}",
    if_exists='replace',
    # project_id=PROJECT_ID
)


# -----------------------------------------------------------------------------
# TEXT + EMBEDDING GENERATION: Recipe Profiles
# -----------------------------------------------------------------------------

class RecipeProfile(BaseModel):
    food_type: str = Field(description="Type of food, e.g., dessert, main course, appetizer")
    cuisine_type: str = Field(description="Cuisine type, e.g., Italian, Chinese, Mexican")
    dietary_preferences: List[str] = Field(description="Dietary preferences, e.g., vegetarian, vegan, gluten-free")
    flavor_profile: List[str] = Field(description="Flavor profile, e.g., spicy, sweet, savory")
    serving_daypart: List[str] = Field(description="Suitable dayparts, e.g., breakfast, lunch, dinner")
    # TODO: add these fields later
    # cooking_method: str = Field(description="Main preparation method (e.g., grilled, baked, stir-fried, raw)")
    # complexity: str = Field(description="Estimated skill or effort required (e.g., easy, intermediate, advanced)")
    # # nutritional_tags: List[str] = Field(description="Health or nutrition tags (e.g., high-protein, low-carb, low-calorie)")
    # # occasion_tags: List[str] = Field(description="Occasions this recipe suits (e.g., everyday meal, party dish, festive special)")
    # ingredient_anchors: List[str] = Field(description="General ingredient families central to the recipe (e.g., poultry, legumes, seafood, grains, leafy greens)")
    
    notes: str = Field(description="Short rationale for the profile")
    justification: str = Field(description="Detailed explanation of how the profile was determined Describe why the food type, cuisine type, dietary preferences, flavor profile, and serving daypart were chosen based on the ingredients and cooking directions. Is not allowed to use quotes or complex punctuation in this field.")


recipe_profile_prompt = f"""Based on the title, ingredients, and cooking directions provided, create a recipe profile that summarizes the key characteristics of this recipe. Your response must follow this exact structure: {schema_to_prompt_with_descriptions(RecipeProfile)}. IMPORTANT: Do not use quotation marks or complex punctuation in your response. Use simple words and avoid any quotes, apostrophes, or special characters."""

recipe_profile_generation_query = f"""
WITH ai_responses AS (
  SELECT 
    s.recipe_id, 
    s.title, 
    s.ingredients, 
    s.cooking_directions, 
    s.nutritions, 
    s.reviews, 
    s.parsed_ingredients, 
    s.parsed_recipe,
    AI.GENERATE(('{recipe_profile_prompt}', s.parsed_ingredients, s.parsed_recipe),
        connection_id => '{CONNECTION_ID}',
        endpoint => 'gemini-2.5-pro',
        model_params => JSON '{{"generationConfig":{{"temperature": 0.0, "maxOutputTokens": 2048, "thinking_config": {{"thinking_budget": 1024}} }} }}',
        output_schema => 'food_type STRING, cuisine_type STRING, dietary_preferences ARRAY<STRING>, flavor_profile ARRAY<STRING>, serving_daypart ARRAY<STRING>, notes STRING, justification STRING'
    ) AS ai_result
  FROM (SELECT * FROM `{RECIPES_PARSED}`) s
)
SELECT 
  *,
  ai_result.full_response AS recipe_profile,
  JSON_EXTRACT_SCALAR(ai_result.full_response, '$.candidates[0].content.parts[0].text') AS recipe_profile_text
FROM ai_responses
"""         #  LIMIT 2

print(recipe_profile_generation_query)

recipe_rows = client.query_and_wait(recipe_profile_generation_query)
df_recipes_profiles = recipe_rows.to_dataframe()

response = json.loads(df_recipes_profiles['recipe_profile'].iloc[0])
r_profile = json.loads(response['candidates'][0]['content']['parts'][0]['text'])
                      
print(json.dumps(response, indent=2))
print(json.dumps(json.loads(df_recipes_profiles['recipe_profile_text'].iloc[0]), indent=2))

df_recipes_profiles.to_gbq(
    destination_table=f"{PROJECT_ID}.{RECIPES_PROFILES_TABLE}",
    if_exists='replace',
    # project_id=PROJECT_ID
)


client.query_and_wait(f"""
ALTER TABLE `{PROJECT_ID}.{RECIPES_PROFILES_TABLE}`
ADD COLUMN text_embedding ARRAY<FLOAT64>
""")

# Create Vector Embeddings for the recipe profiles
client.query_and_wait(f"""
UPDATE `{PROJECT_ID}.{RECIPES_PROFILES_TABLE}` AS t
SET t.text_embedding = s.ml_generate_embedding_result
FROM (
  SELECT
    recipe_id,
    ml_generate_embedding_result
  FROM
    ML.GENERATE_EMBEDDING(
      MODEL `{SCHEMA_NAME}.text_embedding_model`,
      (
        SELECT
          recipe_id,
          recipe_profile_text AS content
        FROM `{PROJECT_ID}.{RECIPES_PROFILES_TABLE}`
      ),
      STRUCT(TRUE AS flatten_json_output)
    )
) AS s
WHERE t.recipe_id = s.recipe_id
""")

# * USERS
# -----------------------------------------------------------------------------
# PARSING: Generate new parsed columns for users during the selected time window into a new table called `users_parsed`
# -----------------------------------------------------------------------------

df_recipe_profiles = client.query_and_wait(f"""SELECT * EXCEPT(text_embedding) FROM `{PROJECT_ID}.{RECIPES_PROFILES_TABLE}`""").to_dataframe()

reviews = []
for idx, row in tqdm(df_recipe_profiles.iterrows(), total=len(df_recipe_profiles)):
    recipe_id = row['recipe_id']
    interactions_dict = ast.literal_eval(row['reviews'])
    for k, v in interactions_dict.items():
        reviews.append({
            'recipe_id': recipe_id,
            'user_id': str(k),
            **v
        })
reviews_df = pd.DataFrame(reviews)
reviews_df.columns = reviews_df.columns.str.lower()
reviews_df['datelastmodified'] = pd.to_datetime(reviews_df['datelastmodified'], format='mixed')

subset_cols = 'user_id, recipe_id, rating, datelastmodified'
df_train_users = client.query_and_wait(f"""SELECT {subset_cols} FROM `{TRAIN_INTERACTIONS}`""").to_dataframe()
df_valid_users = client.query_and_wait(f"""SELECT {subset_cols} FROM `{VALID_INTERACTIONS}`""").to_dataframe()

# Drop users in valid not present in train_set
final_users = set(df_train_users['user_id'].unique()).intersection(set(df_valid_users['user_id'].unique()))
print(f"Final users: {len(final_users)}")
df_train_users = df_train_users[df_train_users['user_id'].isin(final_users)].reset_index(drop=True)
df_train_users['datelastmodified'] = pd.to_datetime(df_train_users['datelastmodified'])
df_train_users = df_train_users.merge(
    reviews_df[['user_id', 'recipe_id', 'datelastmodified', 'text']], how='left', on=['user_id', 'recipe_id', 'datelastmodified']
).rename(columns={'text': 'user_comment'})
df_valid_users = df_valid_users[df_valid_users['user_id'].isin(final_users)].reset_index(drop=True)

print(df_train_users.describe())
print(df_valid_users.describe())

df_users_to_profile = df_valid_users.groupby('user_id').agg({'recipe_id': 'unique'}).reset_index().rename(columns={
    'recipe_id': 'rec_gt'
})  # , 'datelastmodified'


def get_user_history(user_id: int, n: int = 25) -> list:
    """Get the top-n most recent recipes the user has interacted with."""
    user_history = df_train_users[df_train_users['user_id'] == user_id]
    user_history = user_history.sort_values(by='datelastmodified', ascending=False).head(n)
    user_history['date'] = user_history['datelastmodified'].dt.strftime('%Y-%m-%d')

    return user_history[['recipe_id', 'rating', 'date', 'user_comment']].to_dict('records')

    # recipe_ids = user_history['recipe_id'].unique().tolist()
    # return recipe_ids

    # recipes = df_recipes_profiles_to_vs[df_recipes_profiles_to_vs['recipe_id'].isin(recipe_ids)]
    # titles = recipes['title'].tolist()
    # return "\n".join([f"- {t}" for t in titles])


def format_user_history(user_history: list[dict]) -> str:
    """Format the user history as a bulleted list."""

    user_info = ""
    avg_rating = 0
    for entry in user_history:
        recipe_metadata = df_recipe_profiles.loc[lambda df: df['recipe_id'] == entry['recipe_id'], ['recipe_id', 'title', 'parsed_ingredients', 'parsed_recipe', 'recipe_profile_text']].reset_index(drop=True).iloc[0]
        avg_rating += entry['rating']

        user_info += (
            f"\n>>> Recipe Title: {recipe_metadata['title']}\n"
            f">>> User Rating: {entry['rating']}\n"
            f">>> Date of Interaction: {entry['date']}\n\n"
            f">>> User Comment: {entry['user_comment']}\n\n"
            # TODO: Falta esa columnas
            # f"Recipe Average Rating: {row['aver_rate']}\n"
            f">>> Ingredients:\n{recipe_metadata['parsed_ingredients']}\n\n"
            f">>> Cooking Directions:\n{recipe_metadata['parsed_recipe']}\n"
        )
        user_info += "--------------------------------------------\n"
    avg_rating /= len(user_history)
    user_info = f"The user has rated {len(user_history)} recipes, with an average rating of {avg_rating:.2f}.\n{user_info}"
    user_info = "########################################### USER HISTORY START ###########################################\n" + user_info
    user_info += "########################################### USER HISTORY END ###########################################\n"
    
    return user_info


df_users_to_profile['user_history'] = df_users_to_profile['user_id'].apply(get_user_history)
df_users_to_profile['n_history'] = df_users_to_profile['user_history'].apply(len)

df_users_to_profile['n_history'].plot.hist(bins=30)

# print(format_user_history(df_users_to_profile['user_history'].iloc[0]))
df_users_to_profile['history_string'] = df_users_to_profile['user_history'].apply(format_user_history)

bpd.DataFrame(df_users_to_profile).to_gbq(
    destination_table=f"{PROJECT_ID}.{USERS_PARSED}",
    if_exists='replace',
    # project_id=PROJECT_ID
)

# -----------------------------------------------------------------------------
# TEXT + EMBEDDING GENERATION: User Profiles
# -----------------------------------------------------------------------------

class UserProfile(BaseModel):
    liked_cuisines: List[str] = Field(description="List of cuisines the user enjoys most, ranked by preference based on their interaction history and ratings")
    cuisine_preference: str = Field(description="Primary cuisine type the user gravitates towards (e.g., Mediterranean, Asian Fusion, Traditional American)")
    dietary_preference: str = Field(description="Main dietary restriction or lifestyle the user follows (e.g., Vegetarian, Low-carb, No restrictions)")

    food_preferences: List[str] = Field(description="Preferred food categories and meal types (e.g., comfort food, healthy salads, baked goods, grilled meats)")
    cuisine_preferences: List[str] = Field(description="Specific regional or ethnic cuisines the user frequently rates highly (e.g., Thai, Southern BBQ, French pastry)")
    dietary_preferences: List[str] = Field(description="Dietary restrictions, health considerations, or eating patterns (e.g., gluten-free, plant-based, high-protein, dairy-free)")
    flavor_preferences: List[str] = Field(description="Dominant taste profiles and flavor characteristics the user seeks (e.g., bold and spicy, mild and creamy, tangy and citrusy)")
    daypart_preferences: List[str] = Field(description="Preferred times of day for different meal types based on rating patterns (e.g., hearty breakfast, light lunch, elaborate dinner)")
    lifestyle_tags: List[str] = Field(description="Behavioral patterns and cooking style indicators inferred from recipe choices (e.g., quick meals, entertainer, health-conscious, experimental cook)")
    convenience_preference: str = Field(description="Preference for recipe complexity (e.g., quick and easy, gourmet elaborate)")
    diversity_openness: str = Field(description="Willingness to try new cuisines (e.g., adventurous, selective, traditionalist, not defined)")

    notes: str = Field(description="Brief summary explaining the users overall food personality and any notable patterns in their preferences") # . Do not mention specific food names because this is a profile that summarizes the users food personality
    justification: str = Field(description="Detailed explanation of how the profile was determined based on the users interaction history and ratings. Describe why the liked cuisines, cuisine preference, dietary preference, food preferences, cuisine preferences, dietary preferences, flavor preferences, daypart preferences, and lifestyle tags were chosen. Is not allowed to use quotes or complex punctuation in this field. Keep it between 100 and 200 words not more.")


user_profile_prompt = f"""Generate a structured user profile that captures their culinary tastes, dietary preferences, flavor inclinations, among others. Ensure the profile is concise, reasonable and accurately reflects the users food personality based on their interaction history. Please provide a structured profile of the user using the following format: {schema_to_prompt_with_descriptions(UserProfile)}. Each fill of the structured output doesnt need to take more than 200 words keep it in mind. IMPORTANT: Do not use quotation marks or complex punctuation in your response. Use simple words and avoid any quotes, apostrophes, or special characters. Use the following interaction history as reference:"""

user_profile_generation_query = f"""
WITH ai_responses AS (
  SELECT 
    s.user_id, 
    s.n_history,
    s.history_string,
    AI.GENERATE(('{user_profile_prompt}', s.history_string),
        connection_id => '{CONNECTION_ID}',
        endpoint => 'gemini-2.5-flash',
        model_params => JSON '{{"generationConfig":{{"temperature": 1.0, "maxOutputTokens": 4096, "thinking_config": {{"thinking_budget": 1024}} }} }}',
        output_schema => 'liked_cuisines ARRAY<STRING>, cuisine_preference STRING, dietary_preference STRING, food_preferences ARRAY<STRING>, cuisine_preferences ARRAY<STRING>, dietary_preferences ARRAY<STRING>, flavor_preferences ARRAY<STRING>, daypart_preferences ARRAY<STRING>, lifestyle_tags ARRAY<STRING>, notes STRING, justification STRING'
    ) AS ai_result
  FROM (SELECT * FROM `{USERS_PARSED}`) s
)
SELECT 
  *,
  ai_result.full_response AS user_profile,
  JSON_EXTRACT_SCALAR(ai_result.full_response, '$.candidates[0].content.parts[0].text') AS user_profile_text
FROM ai_responses
"""         #   LIMIT 2

print(user_profile_generation_query)
user_rows = client.query_and_wait(user_profile_generation_query)
df_users_profiles = user_rows.to_dataframe()

print(json.dumps(json.loads(df_users_profiles['user_profile_text'].iloc[0]), indent=2))

df_users_profiles.to_gbq(
    destination_table=f"{PROJECT_ID}.{USERS_PROFILES_TABLE}",
    if_exists='replace',
    # project_id=PROJECT_ID
)

client.query_and_wait(f"""
ALTER TABLE `{PROJECT_ID}.{USERS_PROFILES_TABLE}`
ADD COLUMN text_embedding ARRAY<FLOAT64>
""")

# Create Vector Embeddings for the user profiles
client.query_and_wait(f"""
UPDATE `{PROJECT_ID}.{USERS_PROFILES_TABLE}` AS t
SET t.text_embedding = s.ml_generate_embedding_result
FROM (
  SELECT
    user_id,
    ml_generate_embedding_result
  FROM
    ML.GENERATE_EMBEDDING(
      MODEL `{SCHEMA_NAME}.text_embedding_model`,
      (
        SELECT
          user_id,
          user_profile_text AS content
        FROM `{PROJECT_ID}.{USERS_PROFILES_TABLE}`
      ),
      STRUCT(TRUE AS flatten_json_output)
    )
) AS s
WHERE t.user_id = s.user_id
""")

# Parse column to exclude recipe_history from vector search
df = client.query_and_wait(f"""
SELECT * FROM `{PROJECT_ID}.{USERS_PARSED}`
""").to_dataframe()

df['recipes_to_exclude'] = df['user_history'].apply(lambda x: [entry['recipe_id'] for entry in x])

# Update entire table with the new column
df.to_gbq(
    destination_table=f"{PROJECT_ID}.{USERS_PARSED}",
    if_exists='replace',
    # project_id=PROJECT_ID
)

# Add new column to user profiles table via left join
client.query_and_wait(f"""
CREATE OR REPLACE TABLE `{PROJECT_ID}.{USERS_PROFILES_TABLE}` AS
SELECT u.*, p.recipes_to_exclude
FROM `{PROJECT_ID}.{USERS_PROFILES_TABLE}` u
LEFT JOIN `{PROJECT_ID}.{USERS_PARSED}` p USING(user_id)
""")

# -----------------------------------------------------------------------------
# TODO: VECTOR SEARCH & EVAL
# -----------------------------------------------------------------------------

user_queries = client.query_and_wait(f"""
SELECT user_id, recipes_to_exclude, text_embedding FROM `{PROJECT_ID}.{USERS_PROFILES_TABLE}`
""").to_dataframe()
df_recipe_profiles = client.query_and_wait(f"""
  SELECT recipe_id, text_embedding FROM `{PROJECT_ID}.{RECIPES_PROFILES_TABLE}` LIMIT 2
""").to_dataframe()

TOP_K = 50
query_vector_search = f"""
SELECT * FROM
VECTOR_SEARCH(
    (
        SELECT
        title, recipe_id, recipe_profile_text, parsed_ingredients, parsed_recipe, text_embedding
        FROM `{PROJECT_ID}.{RECIPES_PROFILES_TABLE}`
    ),
    'text_embedding',
    (
        SELECT
        text_embedding, n_history, user_id, rec_gt, recipes_to_exclude, user_profile_text
        FROM `{PROJECT_ID}.{USERS_PROFILES_TABLE}`
    ),
    'text_embedding',
    top_k => {TOP_K},
    distance_type => 'EUCLIDEAN'
)
""" # DOT_PRODUCT, COSINE, EUCLIDEAN / LIMIT 2
matches = client.query_and_wait(query_vector_search).to_dataframe()

df_query = pd.json_normalize(matches["query"]).rename(columns={
    'text_embedding': 'user_text_embedding',
})
df_base = pd.json_normalize(matches["base"]).rename(columns={
    'text_embedding': 'recipe_text_embedding',
})
matches = pd.concat([matches.drop(["query","base"], axis=1), df_query, df_base], axis=1)

# Calculate Hit Rate @ K
hit_rate_table = matches.groupby('user_id', as_index=False).agg({
    'rec_gt': 'first', 
    'recipe_id': list,
    'recipe_profile_text': list,
    'title': list,
    'user_profile_text': 'first',   # or unique
    'n_history': 'first'
})
hit_rate_table['hit'] = hit_rate_table.apply(lambda row: any(r in row['rec_gt'] for r in row['recipe_id']), axis=1)
hit_rate_table['hit_count'] = hit_rate_table.apply(lambda row: sum(1 for r in row['rec_gt'] if r in row['recipe_id']), axis=1)
hit_rate_table['hit_proportion'] = hit_rate_table['hit_count'] / hit_rate_table['rec_gt'].apply(len)

avg_hit_prop = hit_rate_table['hit_proportion'].mean()
std_hit_prop = hit_rate_table['hit_proportion'].std()
hit_rate_table['hit_proportion'].plot(kind='hist', bins=20, title=f'Hit Proportion @ {TOP_K} -> {avg_hit_prop:.2f}')
print(f"{avg_hit_prop=:.2f} {std_hit_prop=:.2f}")       # 0.31 until now

display(hit_rate_table.loc[lambda df: df['hit_proportion'] == 0])
hit_rate_table.loc[lambda df: df['hit_proportion'] != 0]['n_history'].hist(bins=20)
hit_rate_table.loc[lambda df: df['hit_proportion'] == 0]['n_history'].hist(bins=20)

# TABLE `{PROJECT_ID}.{RECIPES_PROFILES_TABLE}`,
# WHERE recipe_id NOT IN UNNEST(@excluding_history_recipes_ids)
    # options => JSON '{{"fraction_lists_to_search": 0.01}}'

# https://cloud.google.com/python/docs/reference/bigframes/latest
# https://cloud.google.com/bigquery/docs/samples/bigquery-query#bigquery_query-python