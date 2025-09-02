
import ast
from tqdm import tqdm
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from IPython.display import display
import os, ast, numpy as np, pandas as pd
import random
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


os.chdir(os.path.dirname(os.path.abspath(__file__)))

random.seed(0)

recipes = pd.read_csv('./data/food_recsys/raw-data_recipe.csv')
recipes['recipe_id'] = recipes['recipe_id'].astype(str)
interactions = pd.read_csv('./data/food_recsys/raw-data_interaction.csv')
interactions['user_id'] = interactions['user_id'].astype(str)
interactions['recipe_id'] = interactions['recipe_id'].astype(str)

display('recipes', recipes.head(5))
display('interactions', interactions.head(5))


core_recipes = pd.read_csv('./data/food_recsys/core-data_recipe.csv')


def clean_df(core: pd.DataFrame) -> pd.DataFrame:
    core['user_id'] = core['user_id'].astype(str)
    core['recipe_id'] = core['recipe_id'].astype(str)
    core['dateLastModified'] = core['dateLastModified'].apply(lambda v: v.replace('\n', ''))
    core['dateLastModified'] = pd.to_datetime(core['dateLastModified'], format='ISO8601')
    core['month'] = core['dateLastModified'].dt.month
    core['quarter'] = core['dateLastModified'].dt.quarter
    
    core = core.dropna(how='any')
    return core


core_train_rating = pd.read_csv('./data/food_recsys/core-data-train_rating.csv')
core_train_rating = clean_df(core_train_rating)
display('train', core_train_rating.dtypes, core_train_rating.describe())
print(f"{core_train_rating.shape=}")
core_train_rating['user_id'].value_counts().reset_index().hist(bins=50, log=True)   # .loc[lambda df: df['count'] > 10]['count']


core_test_rating = pd.read_csv('./data/food_recsys/core-data-test_rating.csv')
core_test_rating = clean_df(core_test_rating)
display('test', core_test_rating.dtypes, core_test_rating.describe())


core_val_rating = pd.read_csv('./data/food_recsys/core-data-valid_rating.csv')
core_val_rating = clean_df(core_val_rating)
display('val', core_val_rating.dtypes, core_val_rating.describe())



# -----------------------------------------------------------------------------
# Filter user by minimum interactions in all splits
# -----------------------------------------------------------------------------
K = 5
u_train = core_train_rating['user_id'].value_counts().loc[lambda s: s >= K].index
u_test  = core_test_rating['user_id'].value_counts().loc[lambda s: s >= K].index
u_val   = core_val_rating['user_id'].value_counts().loc[lambda s: s >= K].index

valid_users = set(u_train).intersection(u_test).intersection(u_val)

core_train_rating = core_train_rating[core_train_rating['user_id'].isin(valid_users)]
core_test_rating  = core_test_rating [core_test_rating ['user_id'].isin(valid_users)]
core_val_rating   = core_val_rating  [core_val_rating  ['user_id'].isin(valid_users)]


print(f"Usuarios válidos: {len(valid_users)}")
print(f"{core_train_rating.shape=}")
print(f"{core_test_rating.shape=}")
print(f"{core_val_rating.shape=}")



# -----------------------------------------------------------------------------
# Implicit
# -----------------------------------------------------------------------------
import implicit
from implicit.evaluation import precision_at_k, mean_average_precision_at_k, ndcg_at_k
from scipy.sparse import coo_matrix

user_ids = core_train_rating['user_id'].astype('category')
recipe_ids = core_train_rating['recipe_id'].astype('category')

user_map = dict(enumerate(user_ids.cat.categories))  # idx -> user_id
recipe_map = dict(enumerate(recipe_ids.cat.categories))  # idx -> recipe_id


rows = user_ids.cat.codes.values
cols = recipe_ids.cat.codes.values
data = core_train_rating['rating'].astype(float).values

# Build sparse matrix: users x items
train_matrix = coo_matrix((data, (rows, cols)), shape=(len(user_map), len(recipe_map)))
print("User-Item Matrix:", train_matrix.shape)


# Train ALS model
als_model = implicit.als.AlternatingLeastSquares(
    factors=64,          # latent dims
    regularization=0.01, # L2 reg
    iterations=50,       # ALS steps
    random_state=0
)

als_model.fit(train_matrix.T)


def build_matrix(df, user_ids, recipe_ids):
    # filter to known users and recipes
    df = df[df['user_id'].isin(user_ids.cat.categories) & df['recipe_id'].isin(recipe_ids.cat.categories)]

    rows = df['user_id'].astype('category').cat.set_categories(user_ids.cat.categories).cat.codes
    cols = df['recipe_id'].astype('category').cat.set_categories(recipe_ids.cat.categories).cat.codes
    data = df['rating'].astype(float).values

    return coo_matrix(
        (data, (rows, cols)),
        shape=(len(user_ids.cat.categories), len(recipe_ids.cat.categories))
    )


val_matrix  = build_matrix(core_val_rating, user_ids, recipe_ids)
test_matrix = build_matrix(core_test_rating, user_ids, recipe_ids)


K = 5
# Validation metrics
val_prec = precision_at_k(als_model, train_matrix.T, val_matrix.T, K=K)
val_mapk = mean_average_precision_at_k(als_model, train_matrix.T, val_matrix.T, K=K)
val_ndcg = ndcg_at_k(als_model, train_matrix.T, val_matrix.T, K=K)

print(f"Validation Precision@{K}: {val_prec:.4f}")
print(f"Validation MAP@{K}: {val_mapk:.4f}")
print(f"Validation NDCG@{K}: {val_ndcg:.4f}")

# Test metrics
test_prec = precision_at_k(als_model, train_matrix.T, test_matrix.T, K=K)
test_mapk = mean_average_precision_at_k(als_model, train_matrix.T, test_matrix.T, K=K)
test_ndcg = ndcg_at_k(als_model, train_matrix.T, test_matrix.T, K=K)

print(f"Test Precision@{K}: {test_prec:.4f}")
print(f"Test MAP@{K}: {test_mapk:.4f}")
print(f"Test NDCG@{K}: {test_ndcg:.4f}")



# -----------------------------------------------------------------------------
# Sampling dataset -> ! Working only with `core_val_rating`
# -----------------------------------------------------------------------------
sample_users = random.sample(core_train_rating['user_id'].unique().tolist(), 1000)

train_users = core_train_rating[core_train_rating['user_id'].isin(sample_users)]
val_users = core_val_rating[core_val_rating['user_id'].isin(sample_users)]
test_users = core_test_rating[core_test_rating['user_id'].isin(sample_users)]


train_recipes = recipes[recipes['recipe_id'].isin(train_users['recipe_id'].unique())]
test_recipes = recipes[recipes['recipe_id'].isin(test_users['recipe_id'].unique())]

print("Sampled users:", len(sample_users))
# Calculate if all `sample_users` are present in val and test splits
print("Users in val:", len(val_users['user_id'].unique()))
print("Users in test:", len(test_users['user_id'].unique()))

display(train_users['recipe_id'].value_counts().sort_values(ascending=False))


# -----------------------------------------------------------------------------
# Item Features (Recipes Nutrition)
# -----------------------------------------------------------------------------

all_recipes = pd.concat([train_recipes, test_recipes]).drop_duplicates(subset=['recipe_id'])
nutrition_values = []
for idx, row in tqdm(all_recipes.iterrows(), total=len(all_recipes)):
    nutritions_dict = ast.literal_eval(row['nutritions'])
    
    row_info = {'recipe_id': row['recipe_id']}
    for k in ['niacin', 'sugars', 'sodium', 'carbohydrates', 'vitaminB6', 'calories', 'thiamin', 'fat', 'folate', 'caloriesFromFat', 'calcium', 'fiber', 'magnesium', 'iron', 'cholesterol', 'protein', 'vitaminA', 'potassium', 'saturatedFat', 'vitaminC']:
        if k in nutritions_dict:
            row_info[k] = nutritions_dict[k].get('percentDailyValue', -1)
            if row_info[k] is not None:
                v = str(row_info[k]).strip()
                if v == '< 1':
                    row_info[k] = 0.0
                if v == '-':
                    row_info[k] = -1
                
                try:
                    row_info[k] = float(row_info[k])
                except Exception:
                    row_info[k] = -2
    nutrition_values.append(row_info)

nutrition_df = pd.DataFrame(nutrition_values).fillna(-2)
display(nutrition_df)
print(nutrition_df.dtypes)


def get_user_features(df):
    rows = []
    for idx, v in tqdm(df.groupby('user_id')):
        rows.append({
            'user_id': idx,
            'mean_aver_rate': v['rating'].mean(),
            'count_recipes': v['recipe_id'].nunique(),
            '1_stars_rating_received': (v['rating'] == 1).sum(),
            '2_stars_rating_received': (v['rating'] == 2).sum(),
            '3_stars_rating_received': (v['rating'] == 3).sum(),
            '4_stars_rating_received': (v['rating'] == 4).sum(),
            '5_stars_rating_received': (v['rating'] == 5).sum(),
            'monday_review_count': (v['dateLastModified'].dt.dayofweek == 0).sum(),
            'tuesday_review_count': (v['dateLastModified'].dt.dayofweek == 1).sum(),
            'wednesday_review_count': (v['dateLastModified'].dt.dayofweek == 2).sum(),
            'thursday_review_count': (v['dateLastModified'].dt.dayofweek == 3).sum(),
            'friday_review_count': (v['dateLastModified'].dt.dayofweek == 4).sum(),
            'saturday_review_count': (v['dateLastModified'].dt.dayofweek == 5).sum(),
            'sunday_review_count': (v['dateLastModified'].dt.dayofweek == 6).sum(),
            
            'evening_review_count': ((v['dateLastModified'].dt.hour >= 18) & (v['dateLastModified'].dt.hour <= 23)).sum(),
            # TODO: morning, quarters, etc
        })
    return pd.DataFrame(rows)


def get_product_features(df):
    df = df[['recipe_id', 'aver_rate', 'review_nums']].merge(
        nutrition_df,
        on='recipe_id',
        how='left'
    )
    return df
    
train_X_p = get_product_features(train_recipes)
train_X_u = get_user_features(train_users)
train_y = train_users[['user_id', 'recipe_id', 'rating']]

test_X_p = get_product_features(test_recipes)
test_X_u = get_user_features(test_users)
test_y = test_users[['user_id', 'recipe_id', 'rating']]

print(f"{set(test_recipes['recipe_id'].values).issubset(train_recipes['recipe_id'].values)=}")

def get_model_input(X_u, X_m, y):
    merged = pd.merge(X_u, y, on=['user_id'], how='inner')
    merged = pd.merge(X_m, merged, on=['recipe_id'], how='outer')     # Maintain all records from left dataset even items without ratings
    
    merged.fillna(0, inplace=True)
    features_cols = list(merged.drop(columns=['user_id', 'recipe_id', 'rating']).columns)

    query_list = merged['user_id'].value_counts()
    merged = merged.set_index(['user_id', 'recipe_id'])

    query_list = query_list.sort_index()
    merged.sort_index(inplace=True)

    df_x = merged[features_cols]
    df_y = merged['rating']

    return df_x, df_y, query_list

X_train, y_train, query_list_train = get_model_input(train_X_u, train_X_p, train_y)
X_test, y_test, query_list_test = get_model_input(test_X_u, test_X_p, test_y)


from xgboost import XGBRanker
from sklearn.metrics import ndcg_score

def hit_rate_at_k(y_true, y_pred, k=5):
    """
    Calcula el Hit Rate@K como el solapamiento entre
    el Top-K predicho y el Top-K real.
    
    Parámetros
    ----------
    y_true : array-like
        Ratings verdaderos (se asume que mayor = más relevante)
    y_pred : array-like
        Scores predichos por el modelo
    k : int
        Número de posiciones top-k a considerar
    
    Retorna
    -------
    hit_rate : float
        Proporción de ítems en el Top-K predicho que también están en el Top-K real
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Top-K según el modelo
    top_k_pred = set(np.argsort(y_pred)[::-1][:k])
    
    # Top-K real (ground truth)
    top_k_true = set(np.argsort(y_true)[::-1][:k])
    
    # Intersección
    hits = len(top_k_pred.intersection(top_k_true))
    
    return hits / k


def mean_hit_rate(model, X, y, query_groups, k=5):
    """
    Calcula Hit Rate@K promedio sobre todas las queries.
    
    Parámetros
    ----------
    model : fitted XGBRanker
    X : pd.DataFrame
        Features
    y : pd.Series o array
        Relevancias (ground truth)
    query_groups : list[int]
        Cantidad de ítems por query (misma que usaste en group=)
    k : int
        Top-K
    
    Retorna
    -------
    float
        HitRate@K promedio
    """
    y_pred = model.predict(X)
    results = []
    
    start = 0
    for g in query_groups:
        end = start + g
        y_true_q = y[start:end]
        y_pred_q = y_pred[start:end]
        results.append(hit_rate_at_k(y_true_q, y_pred_q, k))
        start = end
    
    return np.mean(results)


model = XGBRanker(
    objective='rank:ndcg', 
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    random_state=0
)

model.fit(
    X_train,
    y_train,
    group=query_list_train,
    # eval_metric='ndcg',
    eval_set=[(X_test, y_test)],
    eval_group=[list(query_list_test)],
    verbose=10
)

# HitRate@5 en train
train_hit_rate = mean_hit_rate(model, X_train, y_train, query_list_train, k=5)
# HitRate@5 en test
test_hit_rate = mean_hit_rate(model, X_test, y_test, query_list_test, k=5)
print(f"HitRate@5 Train: {train_hit_rate:.4f}")
print(f"HitRate@5 Test:  {test_hit_rate:.4f}")



user_id_sample = '1011486'
pred = model.predict(X_train.loc[user_id_sample])
y_true = y_train.loc[user_id_sample].values
y_pred = pred

# Calculate NDCG@5 and NDCG@10
ndcg_5 = ndcg_score([y_true], [y_pred], k=5)
ndcg_10 = ndcg_score([y_true], [y_pred], k=10)

print(f"User {user_id_sample}:")
print(f"NDCG@5: {ndcg_5:.4f}")
print(f"NDCG@10: {ndcg_10:.4f}")


hit_rate_at_k_5 = hit_rate_at_k(y_true, y_pred, k=5)
print(f"Hit Rate@5: {hit_rate_at_k_5:.4f}")
hit_rate_at_k_10 = hit_rate_at_k(y_true, y_pred, k=10)
print(f"Hit Rate@10: {hit_rate_at_k_10:.4f}")
hit_rate_at_k_20 = hit_rate_at_k(y_true, y_pred, k=20)
print(f"Hit Rate@20: {hit_rate_at_k_20:.4f}")


# Show true ratings and predictions for comparison
print(f"\nTrue ratings: {y_true}")
print(f"Predictions: {y_pred}")


df_compare = pd.DataFrame({
    "true_rating": y_train.loc[user_id_sample].values,
    "pred_score": pred
}, index=y_train.loc[user_id_sample].index).assign(
    spred_rank = lambda df: df['pred_score'].rank(ascending=False, method='min').astype(int)
).sort_values(by='pred_score', ascending=False)
display(df_compare)






# -----------------------------------------------------------------------------
# Item Features (Recipes Ingredients + Directions)
# -----------------------------------------------------------------------------


def prep_ingredients(text):
    if pd.isna(text): return ""
    # Ingredients are caret-separated in your data
    return " ".join(str(text).lower().split('^'))


def prep_directions(text):
    if pd.isna(text): return ""
    s = str(text)
    # Some rows look like dict-strings with 'directions' inside; just fall back to raw text
    # Optionally, try to parse if it starts with "{"
    if s.strip().startswith("{"):
        try:
            d = ast.literal_eval(s)
            # common keys: 'directions' (string) or list
            v = d.get('directions', "")
            if isinstance(v, list):
                return " ".join([str(t) for t in v]).lower()
            return str(v).lower()
        except Exception:
            return s.lower()
    return s.lower()


train_recipes['ingr_txt'] = train_recipes['ingredients'].apply(prep_ingredients)
train_recipes['dir_txt']  = train_recipes['cooking_directions'].apply(prep_directions)


# VibeCoded
ingr_vec = TfidfVectorizer(min_df=5, max_features=3000, ngram_range=(1,2), token_pattern=r"[A-Za-z]{3,}")
dir_vec  = TfidfVectorizer(min_df=5, max_features=2000, ngram_range=(1,2), token_pattern=r"[A-Za-z]{3,}")

X_ingr = ingr_vec.fit_transform(train_recipes['ingr_txt'])
X_dir  = dir_vec.fit_transform(train_recipes['dir_txt'])
# END VibeCoded

vars(ingr_vec)
vars(X_ingr)

vars(dir_vec)
vars(X_dir)


train_recipes = train_recipes.merge(nutrition_df, on='recipe_id', how='left')
num_cols = train_recipes.select_dtypes(include=['float64', 'int64']).columns.tolist()

num_vars = train_recipes[num_cols].fillna(0).to_numpy(dtype=float)
scaler = StandardScaler(with_mean=False)  # sparse-friendly std scaling (no centering)
X_num = scaler.fit_transform(sparse.csr_matrix(num_vars))

# Final item features
item_features = sparse.hstack([X_ingr, X_dir, X_num]).tocsr()

# Map recipe_id -> item index (row in item_features)
recipe2idx = {rid: i for i, rid in enumerate(train_recipes['recipe_id'].tolist())}
idx2recipe = {i: rid for rid, i in recipe2idx.items()}
n_items = len(recipe2idx)

# -----------------------------------------------------------------------------
# User Features
# -----------------------------------------------------------------------------

"""
# TODO: based on the sample user interaction history, generate user profiles first using basic stats and then with LLM
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from langchain.chat_models import init_chat_model

llm = init_chat_model(
    "us.amazon.nova-micro-v1:0",
    # "us.anthropic.claude-sonnet-4-20250514-v1:0",
    model_provider="bedrock_converse",
    temperature=0.0,
)

class UserProfile(BaseModel):
    liked_cuisines: List[str] = Field(description="Given the history of user interactions, list the cuisines the user likes more in order.")
"""

n_users = len(sample_users)
user2idx = {u: i for i, u in enumerate(sample_users)}
idx2user = {i: u for u, i in user2idx.items()}

u_stats = (
    core_train_rating[core_train_rating['user_id'].isin(sample_users)]
    .groupby('user_id')
    .agg(count=('rating', 'size'), mean=('rating', 'mean'))
    .reindex(sample_users)
    .fillna({'count': 0, 'mean': 0.0})
)

# Two minimal features: log(1+count), normalized mean rating
u_counts = np.log1p(u_stats['count'].values).reshape(-1, 1)   # shape = (n_users, 1)
u_mean   = (u_stats['mean'].values / 5.0).reshape(-1, 1)      # shape = (n_users, 1)

user_basic = sparse.csr_matrix(np.hstack([u_counts, u_mean])) # (n_users, 2)
user_identity = sparse.identity(n_users, format="csr")

# Final USER FEATURE MATRIX (identity + basic features)
user_matrix = sparse.hstack([user_identity, user_basic], format="csr").tocsr() # 

print(user_matrix)

# -----------------------------------------------------------------------------
# LightFM Train & Evaluation
# -----------------------------------------------------------------------------

items_univ = list(recipe2idx.keys())
n_items = len(items_univ)

# Helper to build user–item matrices (implicit: rating >= 4)
def make_ui_matrix(df, user2idx, recipe2idx, n_users, n_items, threshold=4):
    df = df[df['user_id'].isin(user2idx) & df['recipe_id'].isin(recipe2idx)].copy()
    if df.empty:
        return sparse.csr_matrix((n_users, n_items))
    rows = df['user_id'].map(user2idx).astype(int).values
    cols = df['recipe_id'].map(recipe2idx).astype(int).values
    data = (df['rating'].astype(float).values >= threshold).astype(np.float32)
    return sparse.coo_matrix((data, (rows, cols)), shape=(n_users, n_items)).tocsr()

train_ui = make_ui_matrix(core_train_rating.loc[lambda df: df['user_id'].isin(sample_users)], user2idx, recipe2idx, n_users, n_items, threshold=4)
test_ui  = make_ui_matrix(core_test_rating.loc[lambda df: df['user_id'].isin(sample_users)],  user2idx, recipe2idx, n_users, n_items, threshold=4)


# Align item features to LightFM expectations: add identity and align order to items_univ
rows_in_item_features = [recipe2idx[rid] for rid in items_univ]
item_features_aligned = item_features[rows_in_item_features, :]
item_identity = sparse.identity(n_items, format="csr")
item_features_final = sparse.hstack([item_identity, item_features_aligned], format="csr").tocsr()

# Train model (WARP works well for implicit feedback)
from lightfm import LightFM
from lightfm.evaluation import precision_at_k as lf_precision_at_k
from lightfm.evaluation import recall_at_k as lf_recall_at_k

model = LightFM(no_components=64, loss='warp', random_state=42)
model.fit(
    interactions=train_ui,
    user_features=user_matrix,
    item_features=item_features_final,
    epochs=15,
    num_threads=4
)


# Evaluate on VAL and TEST (mask seen items by passing train_interactions)
K = 5

test_prec = lf_precision_at_k(model, test_ui, train_interactions=train_ui, k=K,
                           user_features=user_matrix, item_features=item_features_final).mean()
test_rec  = lf_recall_at_k(model,    test_ui, train_interactions=train_ui, k=K,
                        user_features=user_matrix, item_features=item_features_final).mean()
print(f"[TEST] Precision@{K}: {test_prec:.4f} | Recall@{K}: {test_rec:.4f}")



# -----------------------------------------------------------------------------
# GCP
# -----------------------------------------------------------------------------

PROJECT_ID = "hacka-7h2"  # @param {type:"string"}

# # Set the project id
# ! gcloud config set project {PROJECT_ID}
# ! gcloud auth login



"""
IDEA:

1. Tomar un dataset y generar perfiles/descripciones de usuario (basado en historia de interacciones/compras)
2. Generar descripciones de alimentos/productos
3. Hacer un RecSys basico/Replicar uno
4. Generar explicabilidad de cada item post rankeo

https://www.kaggle.com/datasets/elisaxxygao/foodrecsysv1
https://github.com/WUT-IDEA/MealRecPlus

https://github.com/WUT-IDEA/MealRec
https://arxiv.org/abs/2205.12133


https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/applying-llms-to-data/bigquery_generative_ai_intro.ipynb
https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/applying-llms-to-data/bigquery_embeddings_vector_search.ipynb
https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/applying-llms-to-data/multimodal-analysis-bigquery/analyze_multimodal_data_bigquery.ipynb
"""
