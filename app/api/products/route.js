import { executeQuery, getRecipeImageUrl, getFallbackImageUrl, CommonQueries } from '../../lib/bigquery';

export async function GET(request) {
  try {
    // Extract query parameters for filtering
    const { searchParams } = new URL(request.url);
    const category = searchParams.get('category');
    const searchParam = searchParams.get('search');
    const search = searchParam ? searchParam.toLowerCase() : '';
    const userId = searchParams.get('userId') || '4368963'; // Default user ID
    
    let recipes = [];
    
    // If search is provided, use semantic search in recipes table and return 6 results
    if (search && search.trim() !== '') {
      console.log(`Semantic searching recipes for: "${search}"`);
      
      try {
        // Use the semantic search from CommonQueries
        const semanticQuery = CommonQueries.getSemanticSearch(search.trim(), 6);
        recipes = await executeQuery(semanticQuery);
        console.log(recipes)
        console.log(`Found ${recipes.length} semantic search results for "${search}"`);
        
        // If semantic search fails or returns no results, fallback to LIKE search
        if (recipes.length === 0) {
          console.log('Semantic search returned no results, falling back to LIKE search');
          let likeSearchQuery = `
            SELECT 
              recipe_id,
              title as product_name,
              ingredients,
              reviews
            FROM \`kaggle-bigquery-471522.foodrecsys.recipes\`
            WHERE LOWER(title) LIKE '%${search.toLowerCase()}%'
            ORDER BY title
            LIMIT 6
          `;
          recipes = await executeQuery(likeSearchQuery);
          console.log(`Found ${recipes.length} LIKE search results for "${search}"`);
        }
      } catch (semanticError) {
        console.log('Semantic search failed, falling back to LIKE search:', semanticError.message);
        let likeSearchQuery = `
          SELECT 
            recipe_id,
            title as product_name,
            ingredients,
            reviews
          FROM \`kaggle-bigquery-471522.foodrecsys.recipes\`
          WHERE LOWER(title) LIKE '%${search.toLowerCase()}%'
          ORDER BY title
          LIMIT 6
        `;
        recipes = await executeQuery(likeSearchQuery);
        console.log(`Found ${recipes.length} LIKE search fallback results for "${search}"`);
      }
    } else {
      // Default behavior: First try vs_recommendations, then fallback to recipes
      
      // First, try to get products from vs_recommendations where judge_veredict is false
      try {
        let vsQuery = `
          SELECT 
            vs.recipe_id,
            vs.title as product_name,
            vs.user_id,
            vs.judge_veredict,
            r.ingredients,
            r.reviews
          FROM \`kaggle-bigquery-471522.foodrecsys.vs_recommendations\` vs
          LEFT JOIN \`kaggle-bigquery-471522.foodrecsys.recipes\` r ON vs.recipe_id = r.recipe_id
          WHERE vs.user_id = '${userId}' 
            AND vs.judge_veredict = false
          LIMIT 8
        `;
        
        recipes = await executeQuery(vsQuery);
        console.log(`Found ${recipes.length} items from vs_recommendations with judge_veredict=false`);
        
      } catch (vsError) {
        console.log('Error querying vs_recommendations:', vsError.message);
        recipes = []; // Continue to fallback
      }
      
      // If no items found in vs_recommendations, fallback to recipes table
      if (recipes.length === 0) {
        console.log('No items found in vs_recommendations, falling back to recipes table');
        
        let fallbackQuery = `
          SELECT 
            recipe_id,
            title as product_name,
            ingredients,
            reviews
          FROM \`kaggle-bigquery-471522.foodrecsys.recipes\`
          WHERE recipe_id IS NOT NULL
          ORDER BY RAND() 
          LIMIT 9
        `;
        
        recipes = await executeQuery(fallbackQuery);
        console.log(`Found ${recipes.length} items from recipes table as fallback`);
      }
    }

    // Transform data to product catalog format
    let transformedProducts = recipes.map((recipe) => {
      // Generate realistic price based on recipe complexity and random factors
      const basePrice = 12 + Math.random() * 8;
      const price = Math.round(basePrice * 100) / 100;
      
      // Generate image URLs using recipe_id - ALWAYS generate for all products
      const primaryImageUrl = getRecipeImageUrl(recipe.recipe_id);
      const fallbackImageUrl = getFallbackImageUrl(recipe.product_name);
      // Determine category based on title keywords
      const title = (recipe.product_name || '').toLowerCase();
      let productCategory = 'Recipes'; // Default category
      if (title.includes('chicken') || title.includes('beef') || title.includes('pork') || title.includes('meat')) {
        productCategory = 'Meat';
      } else if (title.includes('salad') || title.includes('vegetable') || title.includes('veggie')) {
        productCategory = 'Salads';
      } else if (title.includes('pasta') || title.includes('noodle') || title.includes('spaghetti')) {
        productCategory = 'Pasta';
      } else if (title.includes('soup') || title.includes('stew') || title.includes('broth')) {
        productCategory = 'Soups';
      } else if (title.includes('cake') || title.includes('cookie') || title.includes('dessert') || title.includes('sweet')) {
        productCategory = 'Desserts';
      } else if (title.includes('breakfast') || title.includes('pancake') || title.includes('egg')) {
        productCategory = 'Breakfasts';
      }

      // Process ingredients for display (they might be in different formats)
      let ingredientsList = '';
      if (recipe.ingredients) {
        try {
          // Try to parse if it's a JSON string or handle as plain text
          if (typeof recipe.ingredients === 'string' && recipe.ingredients.includes('^')) {
            // Handle caret-separated ingredients
            ingredientsList = recipe.ingredients.split('^').slice(0, 5).join(', ');
          } else if (typeof recipe.ingredients === 'string') {
            // Handle as plain text, truncate if too long
            ingredientsList = recipe.ingredients.substring(0, 100);
          }
        } catch (e) {
          ingredientsList = 'Ingredients available';
        }
      }

      // Parse reviews to get real rating and review count
      const reviewData = parseReviewsAndGetRating(recipe.reviews);

      return {
        id: recipe.recipe_id,
        recipeId: recipe.recipe_id,
        name: recipe.product_name || 'Recipe',
        description: ingredientsList || generateDescription(recipe.product_name),
        ingredients: recipe.ingredients,
        price: price,
        category: productCategory,
        rating: reviewData.rating,
        reviewCount: reviewData.reviewCount,
        inStock: Math.random() > 0.1, // 90% chance of being in stock
        imageUrl: primaryImageUrl,
        fallbackImageUrl: fallbackImageUrl,
        emoji: getCategoryEmoji(productCategory),
        source: recipe.judge_veredict !== undefined ? 'vs_recommendations' : 'recipes' // Track data source
      };
    });

    // Filter by category if specified
    if (category && category !== 'Todos') {
      transformedProducts = transformedProducts.filter(product => 
        product.category === category
      );
    }

    return Response.json(transformedProducts);
  } catch (error) {
    console.error('Error fetching products:', error);
    
    // Return fallback mock data if BigQuery fails
    const fallbackProducts = [
      {
        id: 'fallback-1',
        name: 'Producto de Ejemplo',
        description: 'Este es un producto de ejemplo mientras se conecta a la base de datos',
        price: 19.99,
        category: 'Ejemplo',
        rating: 4.5,
        inStock: true,
        emoji: '🍽️',
        imageUrl: null,
        fallbackImageUrl: 'https://via.placeholder.com/400x300/6366f1/ffffff?text=Ejemplo'
      }
    ];
    
    return Response.json(fallbackProducts);
  }
}

/**
 * Parse reviews string and calculate average rating using regex
 */
function parseReviewsAndGetRating(reviewsString) {
  if (!reviewsString) {
    return { rating: 4.2, reviewCount: 0 }; // Default values
  }

  try {
    // Use regex to find all 'rating': number patterns
    const ratingMatches = reviewsString.match(/'rating':\s*(\d+)/g);
    
    if (!ratingMatches || ratingMatches.length === 0) {
      return { rating: 4.2, reviewCount: 0 };
    }

    // Extract the actual numbers from the matches
    const ratings = ratingMatches.map(match => {
      const numberMatch = match.match(/(\d+)/);
      return numberMatch ? parseInt(numberMatch[1]) : 0;
    }).filter(rating => rating > 0 && rating <= 5); // Valid ratings only (1-5)

    if (ratings.length === 0) {
      return { rating: 4.2, reviewCount: 0 };
    }

    // Calculate sum and average
    const sum = ratings.reduce((total, rating) => total + rating, 0);
    const averageRating = sum / ratings.length;
    
    return {
      rating: Math.round(averageRating * 10) / 10, // Round to 1 decimal place
      reviewCount: ratings.length
    };

  } catch (error) {
    console.log('Error extracting ratings:', error.message);
    return { rating: 4.2, reviewCount: 0 }; // Fallback values
  }
}

/**
 * Generate a description based on the recipe title
 */
function generateDescription(title) {
  const descriptions = [
    'Deliciosa receta preparada con ingredientes frescos y técnicas tradicionales',
    'Una experiencia culinaria única que deleitará tu paladar',
    'Receta casera con el sabor auténtico que tanto te gusta',
    'Preparación especial con ingredientes de la mejor calidad',
    'Sabores intensos y texturas perfectas en cada bocado',
    'Una combinación perfecta de tradición e innovación culinaria'
  ];
  
  return descriptions[Math.floor(Math.random() * descriptions.length)];
}

/**
 * Get emoji for category
 */
function getCategoryEmoji(category) {
  const emojiMap = {
    'Meat': '🥩',
    'Salads': '🥗',
    'Pasta': '🍝',
    'Soups': '🍲',
    'Desserts': '🍰',
    'Breakfasts': '🥞',
    'Recipes': '🍽️'
  };
  
  return emojiMap[category] || '🍽️';
}


