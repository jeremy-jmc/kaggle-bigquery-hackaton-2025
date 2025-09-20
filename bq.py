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
import matplotlib.pyplot as plt
import numpy as np
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
SUBSET_RECIPE_IDS = f"{PROJECT_ID}.{SCHEMA_NAME}.final_recipes"
SUBSET_USERS_IDS = f"{PROJECT_ID}.{SCHEMA_NAME}.final_users"

RECIPES_ALL = f"{PROJECT_ID}.{SCHEMA_NAME}.recipes"
OUT_DIM = 1024

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

nutrition_values = []
for idx, row in tqdm(df_recipes_pandas.iterrows(), total=len(df_recipes_pandas)):
    nutritions_dict = ast.literal_eval(row['nutritions'])
    
    row_info = {'recipe_id': row['recipe_id']}
    nutritions_info = {}
    for k in ['niacin', 'sugars', 'sodium', 'carbohydrates', 'vitaminB6', 'calories', 'thiamin', 'fat', 'folate', 'caloriesFromFat', 'calcium', 'fiber', 'magnesium', 'iron', 'cholesterol', 'protein', 'vitaminA', 'potassium', 'saturatedFat', 'vitaminC']:
        if k in nutritions_dict:
            nutritions_info[k] = nutritions_dict[k].get('percentDailyValue', -1)
            if nutritions_info[k] is not None:
                v = str(nutritions_info[k]).strip()
                if v == '< 1':
                    nutritions_info[k] = 0.0
                # if v == '-':
                #     nutritions_info[k] = -1
                
                try:
                    nutritions_info[k] = f"{nutritions_info[k]} percent"
                except Exception:
                    # nutritions_info[k] = "-1 %"
                    pass
    
    row_info['percent_daily_values'] = "\n".join([f"{k}: {v}" for k, v in nutritions_info.items()])
    nutrition_values.append(row_info)

nutrition_df = pd.DataFrame(nutrition_values).fillna(-2)

df_recipes_pandas['parsed_ingredients'] = df_recipes_pandas['ingredients'].apply(prep_ingredients)
df_recipes_pandas['parsed_recipe'] = df_recipes_pandas['cooking_directions'].apply(prep_directions)
df_recipes_pandas = df_recipes_pandas.merge(nutrition_df, how='left', on='recipe_id')
df_recipes = bpd.DataFrame(df_recipes_pandas)

# Upload the new table in BigQuery
df_recipes.to_gbq(
    destination_table=f"{PROJECT_ID}.{RECIPES_PARSED}",
    if_exists='replace',
)


# -----------------------------------------------------------------------------
# TEXT + EMBEDDING GENERATION: Recipe Profiles
# -----------------------------------------------------------------------------

class RecipeProfile(BaseModel):
    food_type: str = Field(description="Type of food, e.g., dessert, main course, appetizer")
    cuisine_type: str = Field(description="Cuisine type, e.g., Italian, Chinese, Mexican, American")
    dietary_preferences: List[str] = Field(description="Dietary preferences, e.g., omnivore, vegetarian, vegan, gluten-free")
    flavor_profile: List[str] = Field(description="Flavor profile, e.g., spicy, sweet, savory")
    serving_daypart: List[str] = Field(description="Suitable dayparts, e.g., breakfast, lunch, dinner")
    
    # cooking_method: str = Field(description="Main preparation method (e.g., grilled, baked, stir-fried, raw)")
    # complexity: str = Field(description="Estimated skill or effort required (e.g., easy, intermediate, advanced)")
    # ingredient_anchors: List[str] = Field(description="General ingredient families central to the recipe (e.g., poultry, legumes, seafood, grains, leafy greens)")
    # nutritional_tags: List[str] = Field(description="Health or nutrition tags (e.g., high-protein, low-carb, low-calorie)")
    # occasion_tags: List[str] = Field(description="Occasions this recipe suits (e.g., everyday meal, party dish, festive special)")
    # cooking_time_category: str = Field(description="Qualitative cooking time category inferred from directions, e.g., quick, moderate, long")
    # equipment_needed: List[str] = Field(description="Key kitchen tools inferred from directions, e.g., oven, blender, slow cooker")
    # allergen_risks: List[str] = Field(description="Possible allergens present in the recipe, e.g., nuts, dairy, shellfish, gluten")
    # sensory_descriptors: List[str] = Field(description="Textural or sensory aspects, e.g., crispy, creamy, hearty, light")

    notes: str = Field(description="Short rationale summarizing the recipe profile")
    target_audience: str = Field(description="Types of users who would likely enjoy this recipe based on cooking skill level, flavor intensity, dietary needs, and lifestyle preferences. Helps recommendation systems match recipes to appropriate user profiles.")
    justification: str = Field(description="Detailed explanation of how the profile was determined Describe why the food type, cuisine type, dietary preferences, flavor profile, and serving daypart were chosen based on the ingredients and cooking directions. Is not allowed to use quotes or complex punctuation in this field.")


recipe_profile_prompt = f"""Based on the title, ingredients, cooking directions and percent daily values provided, create a recipe profile that summarizes the key characteristics of this recipe. Your response must follow this exact structure: {schema_to_prompt_with_descriptions(RecipeProfile)}. IMPORTANT: Do not use quotation marks or complex punctuation in your response. Use simple words and avoid any quotes, apostrophes, or special characters."""

# cooking_method STRING, complexity STRING, ingredient_anchors STRING, nutritional_tags STRING, occasion_tags STRING, cooking_time_category STRING, equipment_needed STRING, allergen_risks STRING, sensory_descriptors STRING
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
    AI.GENERATE(('{recipe_profile_prompt}', s.parsed_ingredients, s.parsed_recipe, s.percent_daily_values),
        connection_id => '{CONNECTION_ID}',
        endpoint => 'gemini-2.5-flash',
        model_params => JSON '{{"generationConfig":{{"temperature": 0.5, "maxOutputTokens": 2048, "thinking_config": {{"thinking_budget": 512}} }} }}',
        output_schema => 'food_type STRING, cuisine_type STRING, dietary_preferences ARRAY<STRING>, flavor_profile ARRAY<STRING>, serving_daypart ARRAY<STRING>, notes STRING, target_audience STRING, justification STRING'
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
      STRUCT(TRUE AS flatten_json_output, {OUT_DIM} AS OUTPUT_DIMENSIONALITY, 'RETRIEVAL_DOCUMENT' AS task_type)
    )
) AS s
WHERE t.recipe_id = s.recipe_id
""")

# * USERS
# -----------------------------------------------------------------------------
# PARSING: Generate new parsed columns for users during the selected time window into a new table called `users_parsed`
# -----------------------------------------------------------------------------

df_recipe_metadata = client.query_and_wait(f"""SELECT recipe_id, title, parsed_ingredients, parsed_recipe, recipe_profile_text, reviews FROM `{RECIPES_PROFILES_TABLE}`""").to_dataframe()

reviews = []
for idx, row in tqdm(df_recipe_metadata.iterrows(), total=len(df_recipe_metadata)):
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

SUBSET_COLS = 'user_id, recipe_id, rating, datelastmodified'
df_train_users = client.query_and_wait(f"""SELECT {SUBSET_COLS} FROM `{TRAIN_INTERACTIONS}`""").to_dataframe()
df_valid_users = client.query_and_wait(f"""SELECT {SUBSET_COLS} FROM `{VALID_INTERACTIONS}`""").to_dataframe()

# Drop users in valid not present in train_set
FINAL_USERS = set(df_train_users['user_id'].unique()).intersection(set(df_valid_users['user_id'].unique()))
print(f"Final users: {len(FINAL_USERS)}")

df_users_history = df_train_users[df_train_users['user_id'].isin(FINAL_USERS)].reset_index(drop=True)
df_users_history['datelastmodified'] = pd.to_datetime(df_users_history['datelastmodified'])
df_users_history = df_users_history.merge(
    reviews_df[['user_id', 'recipe_id', 'datelastmodified', 'text']], how='left', 
    on=['user_id', 'recipe_id', 'datelastmodified'],
    validate='one_to_one'
).rename(columns={'text': 'user_comment'})

df_valid_users = df_valid_users[df_valid_users['user_id'].isin(FINAL_USERS)].reset_index(drop=True)

print(df_users_history.describe())
print(df_valid_users.describe())

df_users_to_profile = df_valid_users.groupby('user_id').agg({'recipe_id': 'unique'}).reset_index().rename(columns={
    'recipe_id': 'rec_gt'
})  # , 'datelastmodified'


def get_user_history(user_id: int, n: int = 25, k_min: int = 5) -> list:
    """Get the top-n most recent recipes the user has interacted with."""
    user_history = df_users_history[df_users_history['user_id'] == user_id]
    user_history = user_history.sort_values(by='datelastmodified', ascending=False).head(n)
    assert len(user_history) >= k_min, f"User {user_id} has less than {k_min} interactions in the training set."

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
        recipe_metadata = df_recipe_metadata.loc[lambda df: df['recipe_id'] == entry['recipe_id'], ['recipe_id', 'title', 'parsed_ingredients', 'parsed_recipe', 'recipe_profile_text']].reset_index(drop=True).iloc[0]
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
)

# -----------------------------------------------------------------------------
# TEXT + EMBEDDING GENERATION: User Profiles
# -----------------------------------------------------------------------------

class UserProfile(BaseModel):
    liked_cuisines: List[str] = Field(description="List of cuisines the user enjoys most, ranked by preference based on their interaction history and ratings")
    cuisine_preference: str = Field(description="Primary cuisine type the user gravitates towards (e.g., Mediterranean, Asian Fusion, Traditional American)")
    dietary_preference: str = Field(description="Main dietary restriction or lifestyle the user follows (e.g., Vegetarian, Low-carb, No restrictions)")

    food_preferences: List[str] = Field(description="Preferred food categories and meal types (e.g., comfort food, healthy salads, baked goods, grilled meats)")
    top_cuisine_choices: List[str] = Field(description="Specific regional or ethnic cuisines the user frequently rates highly (e.g., Thai, Southern BBQ, French pastry)")
    dietary_preferences: List[str] = Field(description="Dietary restrictions, health considerations, or eating patterns (e.g., gluten-free, plant-based, high-protein, dairy-free)")
    flavor_preferences: List[str] = Field(description="Dominant taste profiles and flavor characteristics the user seeks (e.g., bold and spicy, mild and creamy, tangy and citrusy)")
    daypart_preferences: List[str] = Field(description="Preferred times of day for different meal types based on rating patterns (e.g., hearty breakfast, light lunch, elaborate dinner)")
    lifestyle_tags: List[str] = Field(description="Behavioral patterns and cooking style indicators inferred from recipe choices (e.g., quick meals, entertainer, health-conscious, experimental cook)")
    # adventurousness_level: str = Field(description="Exploration tendency, e.g., experimental, conservative eater")
    convenience_preference: str = Field(description="Preference for recipe complexity (e.g., quick and easy, gourmet elaborate)")
    diversity_openness: str = Field(description="Willingness to try new cuisines (e.g., adventurous, selective, traditionalist, not defined)")

    notes: str = Field(description="Brief summary explaining the users overall food personality and any notable patterns in their preferences") # . Do not mention specific food names because this is a profile that summarizes the users food personality
    justification: str = Field(description="Detailed explanation of how the profile was determined based on the users interaction history and ratings. Describe why the liked cuisines, cuisine preference, dietary preference, food preferences, cuisine preferences, dietary preferences, flavor preferences, daypart preferences, and lifestyle tags were chosen. Is not allowed to use quotes or complex punctuation in this field. Keep it between 100 and 200 words not more.")
    user_story: str = Field(description="Predictive narrative about the user s culinary evolution and potential future preferences. Describes their food journey, emerging patterns, and likely directions for taste exploration. Written to help predict what they might enjoy next based on their current trajectory and evolving palate.")
    future_preferences: str = Field(description="Speculative insights into the types of recipes and cuisines the user may be inclined to explore in the future. Based on their current preferences, suggest new food categories, cooking styles, or dietary trends they might be open to trying next. This helps in anticipating their evolving culinary interests.")


user_profile_prompt = f"""Generate a structured user profile that captures their culinary tastes, dietary preferences, flavor inclinations, among others. This user profile will be used then for a Recommendation System. Ensure the profile is concise, reasonable and accurately reflects the users food personality based on their interaction history. Please provide a structured profile of the user using the following format: {schema_to_prompt_with_descriptions(UserProfile)}. Each fill of the structured output doesnt need to take more than 200 words keep it in mind. IMPORTANT: Do not use quotation marks or complex punctuation in your response. Use simple words and avoid any quotes, apostrophes, or special characters. Use the following interaction history as reference:"""

# adventurousness_level STRING,
user_profile_generation_query = f"""
WITH ai_responses AS (
  SELECT 
    s.user_id, 
    s.n_history,
    s.history_string,
    AI.GENERATE(('{user_profile_prompt}', s.history_string),
        connection_id => '{CONNECTION_ID}',
        endpoint => 'gemini-2.5-flash',
        model_params => JSON '{{"generationConfig":{{"temperature": 0.5, "maxOutputTokens": 2048, "thinking_config": {{"thinking_budget": 512}} }} }}',
        output_schema => 'liked_cuisines ARRAY<STRING>, cuisine_preference STRING, dietary_preference STRING, food_preferences ARRAY<STRING>, top_cuisine_choices ARRAY<STRING>, dietary_preferences ARRAY<STRING>, flavor_preferences ARRAY<STRING>, daypart_preferences ARRAY<STRING>, lifestyle_tags ARRAY<STRING>, convenience_preference STRING, diversity_openness STRING, notes STRING, justification STRING, user_story STRING, future_preferences STRING'
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
      STRUCT(TRUE AS flatten_json_output, {OUT_DIM} AS OUTPUT_DIMENSIONALITY, 'RETRIEVAL_QUERY' AS task_type)
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
)

# Add new column to user profiles table via left join
client.query_and_wait(f"""
CREATE OR REPLACE TABLE `{PROJECT_ID}.{USERS_PROFILES_TABLE}` AS
SELECT u.*, p.recipes_to_exclude, p.rec_gt
FROM `{PROJECT_ID}.{USERS_PROFILES_TABLE}` u
LEFT JOIN `{PROJECT_ID}.{USERS_PARSED}` p USING(user_id)
""")

# -----------------------------------------------------------------------------
# VECTOR SEARCH
# -----------------------------------------------------------------------------
TOP_K = 20
N_NEIGHBORS = 20

user_queries = client.query_and_wait(f"""
SELECT user_id, recipes_to_exclude, text_embedding FROM `{PROJECT_ID}.{USERS_PROFILES_TABLE}`
""").to_dataframe()
user_queries['len'] = user_queries['text_embedding'].apply(len)
print(f"{user_queries['len'].value_counts()=}")

df_recipe_profiles = client.query_and_wait(f"""
  SELECT recipe_id, recipe_profile_text, text_embedding FROM `{PROJECT_ID}.{RECIPES_PROFILES_TABLE}`
""").to_dataframe()
df_recipe_profiles['len'] = df_recipe_profiles['text_embedding'].apply(len)
print(f"{df_recipe_profiles['len'].value_counts()=}")

excluded_recipes = df_recipe_profiles[df_recipe_profiles['len'] == 0]['recipe_id'].tolist()

query_vector_search = f"""
SELECT * FROM
VECTOR_SEARCH(
    (
        SELECT
        title, recipe_id, recipe_profile_text, parsed_ingredients, parsed_recipe, text_embedding
        FROM `{PROJECT_ID}.{RECIPES_PROFILES_TABLE}`
        WHERE text_embedding IS NOT NULL 
        AND ARRAY_LENGTH(text_embedding) > 0
    ),
    'text_embedding',
    (
        SELECT
        text_embedding, n_history, user_id, rec_gt, recipes_to_exclude, user_profile_text
        FROM `{PROJECT_ID}.{USERS_PROFILES_TABLE}`
        WHERE text_embedding IS NOT NULL 
        AND ARRAY_LENGTH(text_embedding) > 0
    ),
    'text_embedding',
    top_k => {TOP_K},
    distance_type => 'COSINE'
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
hit_rate_table['rec_gt'] = hit_rate_table['rec_gt'].apply(lambda x: [v for v in x if v not in excluded_recipes])
hit_rate_table['hit_count'] = hit_rate_table.apply(lambda row: sum(1 for r in row['rec_gt'] if r in row['recipe_id']), axis=1)
hit_rate_table['hit'] = hit_rate_table['hit_count'] > 0
hit_rate_table['hit_proportion'] = hit_rate_table['hit_count'] / hit_rate_table['rec_gt'].apply(len)

avg_hit_prop = hit_rate_table['hit_proportion'].mean()
std_hit_prop = hit_rate_table['hit_proportion'].std()
print(f"GEMINI EMBEDDINGS {avg_hit_prop=:.2f} {std_hit_prop=:.2f}")       # 0.32 until now

plot = True
if plot:
    hit_rate_table['hit_proportion'].plot(kind='hist', bins=20, title=f'Hit Proportion @ {TOP_K} -> {avg_hit_prop:.5f}')
    plt.show()

# -----------------------------------------------------------------------------
# * MATRIX FACTORIZATION 
# -----------------------------------------------------------------------------
import implicit
from implicit.evaluation import precision_at_k, mean_average_precision_at_k, ndcg_at_k
from implicit.nearest_neighbours import bm25_weight, tfidf_weight
from scipy.sparse import coo_matrix

df_train_users = client.query_and_wait(f"""
SELECT {SUBSET_COLS} FROM `{TRAIN_INTERACTIONS}`
WHERE user_id IN (SELECT user_id FROM `{PROJECT_ID}.{USERS_PARSED}`)
""").to_dataframe()

df_valid_users = client.query_and_wait(f"""
SELECT {SUBSET_COLS} FROM `{VALID_INTERACTIONS}`
WHERE user_id IN (SELECT user_id FROM `{PROJECT_ID}.{USERS_PARSED}`)
""").to_dataframe()

user_ids = df_train_users['user_id'].astype('category')
recipe_ids = df_train_users['recipe_id'].astype('category')

user_map = dict(enumerate(user_ids.cat.categories))  # idx -> user_id
recipe_map = dict(enumerate(recipe_ids.cat.categories))  # idx -> recipe_id

rows = user_ids.cat.codes.values
cols = recipe_ids.cat.codes.values

data = df_train_users['rating'].astype(float).values

# Build sparse matrix: users x items
train_matrix = coo_matrix((data, (rows, cols)), shape=(len(user_map), len(recipe_map)))
print("[TRAIN] User-Item Matrix:", train_matrix.shape)
print("[TRAIN] Non-zero elements:", train_matrix.nnz)
print("[TRAIN] Non-zero proportion:", train_matrix.nnz / (train_matrix.shape[0] * train_matrix.shape[1]))

als_model = implicit.als.AlternatingLeastSquares(
    factors=32,          # latent dims
    regularization=0.05, # L2 reg
    iterations=500,       # ALS steps
    alpha=2.0,          # confidence scaling
    calculate_training_loss=True,
    random_state=0
)

als_model.fit(train_matrix.T)


def get_recommendations(user_id: str, N: int = TOP_K, model = als_model):
    if user_id not in user_ids.cat.categories:
        raise ValueError(f"User ID {user_id} not found in training data.")
    
    user_idx = int(user_ids.cat.categories.get_loc(user_id))
    user_items = train_matrix.tocsr()[user_idx]
    
    recs = model.recommend(
        user_idx,
        user_items,
        N=N,
        filter_already_liked_items=True,
        recalculate_user=False
    )
    
    # recommended_recipes = [(recipe_map[i], score) for i, score in zip(*recs)]
    recommended_recipes = [recipe_map[r_i] for r_i, score in zip(*recs)]
    return recommended_recipes

df_matches_als = (
    df_valid_users.loc[lambda df: df['user_id'].isin(df_train_users['user_id'].values)]
    .groupby('user_id', as_index=False)
    .agg({'recipe_id': list})
).rename(columns={'recipe_id': 'rec_gt'})
df_matches_als['rec_gt'] = df_matches_als['rec_gt'].apply(lambda x: [v for v in x if v not in excluded_recipes])

df_matches_als['als_recommendations'] = df_matches_als['user_id'].apply(get_recommendations)
df_matches_als['hit_count'] = df_matches_als.apply(lambda row: sum(1 for r in row['rec_gt'] if r in row['als_recommendations']), axis=1)
df_matches_als['hit_proportion'] = df_matches_als['hit_count'] / df_matches_als['rec_gt'].apply(len)

avg_hit_prop_als = df_matches_als['hit_proportion'].mean()
std_hit_prop_als = df_matches_als['hit_proportion'].std()

print(f"ALS {avg_hit_prop_als=:.2f} {std_hit_prop_als=:.2f}")

if plot:
    df_matches_als['hit_proportion'].plot(kind='hist', bins=20, title=f'Hit Proportion @ {TOP_K} -> {avg_hit_prop_als:.5f}')
    plt.show()

# https://cloud.google.com/python/docs/reference/bigframes/latest
# https://cloud.google.com/bigquery/docs/samples/bigquery-query#bigquery_query-python