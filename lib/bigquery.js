import { BigQuery } from '@google-cloud/bigquery';

let bigQueryClient = null;

/**
 * Initialize and return a BigQuery client instance
 * Supports both service account file and environment variables authentication
 */
export function getBigQueryClient() {
  if (!bigQueryClient) {
    const options = {
      projectId: process.env.GOOGLE_CLOUD_PROJECT_ID,
    };

    // Try to use service account file first
    if (process.env.GOOGLE_APPLICATION_CREDENTIALS) {
      options.keyFilename = process.env.GOOGLE_APPLICATION_CREDENTIALS;
    }
    // Fallback to environment variables for credentials
    else if (process.env.GOOGLE_CLOUD_PRIVATE_KEY && process.env.GOOGLE_CLOUD_CLIENT_EMAIL) {
      options.credentials = {
        private_key: process.env.GOOGLE_CLOUD_PRIVATE_KEY.replace(/\\n/g, '\n'),
        client_email: process.env.GOOGLE_CLOUD_CLIENT_EMAIL,
      };
    }

    bigQueryClient = new BigQuery(options);
  }

  return bigQueryClient;
}

/**
 * Execute a BigQuery SQL query and return results
 * @param {string} query - The SQL query to execute
 * @param {Object} options - Optional query options
 * @returns {Promise<Array>} Query results
 */
export async function executeQuery(query, options = {}) {
  try {
    const bigquery = getBigQueryClient();
    
    console.log('Executing BigQuery query:', query);
    
    const [job] = await bigquery.createQueryJob({
      query,
      location: 'US', // Change if your dataset is in a different location
      ...options,
    });

    console.log(`Job ${job.id} started.`);

    // Wait for the query to finish
    const [rows] = await job.getQueryResults();
    
    console.log(`Query completed. Retrieved ${rows.length} rows.`);
    
    return rows;
  } catch (error) {
    console.error('BigQuery error:', error);
    throw new Error(`BigQuery query failed: ${error.message}`);
  }
}

/**
 * Get dataset and table configuration from environment variables
 */
export function getTableConfig() {
  return {
    projectId: process.env.GOOGLE_CLOUD_PROJECT_ID,
    datasetId: process.env.BIGQUERY_DATASET_ID,
    tables: {
      products: process.env.BIGQUERY_PRODUCTS_TABLE || 'products',
      users: process.env.BIGQUERY_USERS_TABLE || 'users',
      orders: process.env.BIGQUERY_ORDERS_TABLE || 'orders',
      recommendations: process.env.BIGQUERY_RECOMMENDATIONS_TABLE || 'recommendations',
    },
  };
}

/**
 * Helper function to build fully qualified table names
 * @param {string} tableName - The table name
 * @returns {string} Fully qualified table name (project.dataset.table)
 */
export function getFullTableName(tableName) {
  const config = getTableConfig();
  const tableMap = {
    'user_profiles': 'user_profiles',
    'users_parsed': 'users_parsed',
    'recipes': 'recipes',
    'vs_recommendations': 'vs_recommendations',
    'users': config.tables.users,
    'products': config.tables.products,
    'orders': config.tables.orders,
    'recommendations': config.tables.recommendations
  };
  
  const actualTableName = tableMap[tableName] || tableName;
  return `${config.projectId}.${config.datasetId}.${actualTableName}`;
}

/**
 * Test the BigQuery connection
 * @returns {Promise<boolean>} True if connection is successful
 */
export async function testConnection() {
  try {
    const bigquery = getBigQueryClient();
    
    // Simple query to test connection
    const query = 'SELECT 1 as test';
    const [rows] = await bigquery.query({ query });
    
    console.log('BigQuery connection test successful:', rows);
    return true;
  } catch (error) {
    console.error('BigQuery connection test failed:', error);
    return false;
  }
}

/**
 * Generate Google Cloud Storage image URL for recipe
 * @param {string} recipeId - The recipe ID
 * @returns {string} Full URL to the recipe image
 */
export function getRecipeImageUrl(recipeId) {
  if (!recipeId) return null;
  
  const bucketName = 'kaggle-recipes';
  // Google Cloud Storage public URL format
  return `https://storage.googleapis.com/${bucketName}/core-data-images/core-data-images/${recipeId}.jpg`;
}

/**
 * Generate fallback image URL if recipe image doesn't exist
 * @param {string} title - Recipe title for generating placeholder
 * @returns {string} Fallback image URL
 */
export function getFallbackImageUrl(title) {
  if (!title) return 'https://via.placeholder.com/400x300/6366f1/ffffff?text=Recipe';
  
  // Generate a food-themed placeholder based on title
  const encodedTitle = encodeURIComponent(title);
  return `https://via.placeholder.com/400x300/6366f1/ffffff?text=${encodedTitle}`;
}

/**
 * Common queries for different data types
 */
export const CommonQueries = {
  // Get user profile data from user_profiles table
  getUserProfile: (userId) => `
    SELECT 
      user_id,
      n_history,
      history_string,
      ai_result.convenience_preference,
      ai_result.cuisine_preference,
      ai_result.cuisine_preferences,
      ai_result.daypart_preferences,
      ai_result.dietary_preference,
      ai_result.dietary_preferences,
      ai_result.diversity_openness,
      ai_result.flavor_preferences,
      ai_result.food_preferences,
      ai_result.full_response,
      ai_result.future_preferences,
      ai_result.justification,
      ai_result.lifestyle_tags,
      ai_result.liked_cuisines,
      ai_result.notes,
      ai_result.status,
      ai_result.user_story,
      user_profile,
      user_profile_text,
      text_embedding,
      recipes_to_exclude,
      rec_gt
    FROM ${getFullTableName('user_profiles')}
    WHERE user_id = @userId
  `,

  // Get user orders from users_parsed table with recipe details
  getUserOrders: (userId) => `
    SELECT 
      history.date as order_date,
      history.rating as rating,
      history.user_comment as comment,
      history.recipe_id,
      r.title as product_name
    FROM ${getFullTableName('users_parsed')} up,
    UNNEST(up.user_history) as history
    LEFT JOIN ${getFullTableName('recipes')} r ON history.recipe_id = r.recipe_id
    WHERE up.user_id = @userId
    ORDER BY history.date DESC
  `,

  // Get product catalog
  getProducts: () => `
    SELECT 
      id, name, description, price, category, rating, in_stock, image_url
    FROM ${getFullTableName('products')}
    WHERE active = true
    ORDER BY category, name
  `,

  // Get recommendations for a user from vs_recommendations table
  getRecommendations: (userId) => `
    SELECT 
      recipe_id,
      title,
      ai_result.justification
    FROM ${getFullTableName('vs_recommendations')}
    WHERE user_id = @userId
    AND judge_veredict = true
    ORDER BY RAND()
    LIMIT 6
  `,

    // Get recommendations for a user from vs_recommendations table
  getRecommendationsAdmin: (userId) => `
    SELECT 
      recipe_id,
      title,
      ai_result.justification
    FROM ${getFullTableName('vs_recommendations')}
    WHERE user_id = @userId
    AND judge_veredict = true
    ORDER BY RAND()
    LIMIT 6
  `,

  // Semantic search for recipes using vector embeddings
  getSemanticSearch: (searchQuery, limit = 6) => `
    WITH search_embedding AS (
      SELECT 
        text_embedding AS search_vector
      FROM ML.GENERATE_TEXT_EMBEDDING(
        MODEL \`kaggle-bigquery-471522.foodrecsys.text_embedding_model\`,
        (SELECT '${searchQuery.replace(/'/g, "''")}' AS content),
        STRUCT(1024 as output_dimensionality)
      )
    ),
    recipe_similarities AS (
      SELECT 
        rp.recipe_id,
        rp.title AS product_name,
        rp.ingredients,
        rp.recipe_profile_text,
        r.reviews,
        ML.DISTANCE(rp.text_embedding, s.search_vector, 'COSINE') AS similarity_score
      FROM \`kaggle-bigquery-471522.foodrecsys.recipe_profiles\` rp
      LEFT JOIN \`kaggle-bigquery-471522.foodrecsys.recipes\` r ON rp.recipe_id = r.recipe_id
      CROSS JOIN search_embedding s
      WHERE rp.text_embedding IS NOT NULL
        AND s.search_vector IS NOT NULL
        AND ARRAY_LENGTH(rp.text_embedding) = 1024
      ORDER BY similarity_score ASC
      LIMIT ${limit * 2}
    ),
    filtered_similarities AS (
      SELECT 
        recipe_id,
        product_name,
        ingredients,
        recipe_profile_text,
        reviews,
        similarity_score
      FROM recipe_similarities
      WHERE similarity_score IS NOT NULL 
      ORDER BY similarity_score ASC
      LIMIT ${limit}
    )
    SELECT 
      recipe_id,
      product_name,
      ingredients,
      recipe_profile_text,
      reviews,
      similarity_score
    FROM filtered_similarities;

  `
};