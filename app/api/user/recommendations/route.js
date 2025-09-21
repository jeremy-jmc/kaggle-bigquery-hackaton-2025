import { executeQuery, CommonQueries, getRecipeImageUrl, getFallbackImageUrl } from '../../../lib/bigquery';

export async function GET(request) {
  try {
    // Extract user ID from query parameters or use default
    const { searchParams } = new URL(request.url);
    const userId = searchParams.get('userId') || '4368963';

    // Execute BigQuery to get recommendations
    const recommendations = await executeQuery(
      CommonQueries.getRecommendations(userId),
      {
        params: { userId: userId }
      }
    );

    // Transform data to match frontend expectations
    const transformedData = recommendations.map((rec, index) => {
      // Replace "User prefers" with "Based on your preferences" in justification
      let processedJustification = rec.justification || 'Recommended based on your taste profile';
      processedJustification = processedJustification.replace(/User prefers/gi, 'You show a preference for');
      
      // Generate image URLs
      const primaryImageUrl = getRecipeImageUrl(rec.recipe_id);
      const fallbackImageUrl = getFallbackImageUrl(rec.title);
      
      return {
        id: rec.recipe_id,
        recipeId: rec.recipe_id, // Store for future use
        product: rec.title,
        reason: processedJustification,
        confidence: Math.floor(Math.random() * 25) + 75, // Generate confidence 75-99%
        price: generateRandomPrice(), // Generate realistic price
        description: generateDescription(rec.title),
        rating: generateRandomRating(),
        emoji: getProductEmoji(rec.title),
        imageUrl: primaryImageUrl,
        fallbackImageUrl: fallbackImageUrl
      };
    });

    return Response.json(transformedData);
  } catch (error) {
    console.error('Error fetching recommendations:', error);
    
    // Fallback to mock data if BigQuery fails
    const fallbackData = [
      { 
        id: 1,
        recipeId: "rec_001",
        product: "Risotto de mariscos", 
        reason: "Based on your preferences for seafood and pasta dishes",
        confidence: 85,
        price: 24.99,
        description: "Delicioso risotto con mariscos frescos",
        rating: 4.5,
        emoji: "🦐"
      },
      { 
        id: 2,
        recipeId: "rec_002",
        product: "Salmón a la plancha", 
        reason: "Based on your preferences for healthy fish preparations",
        confidence: 78,
        price: 28.50,
        description: "Salmón fresco a la plancha con vegetales",
        rating: 4.7,
        emoji: "🐟"
      }
    ];
    
    return Response.json(fallbackData);
  }
}

// Helper function to generate realistic prices based on dish type
function generateRandomPrice() {
  const basePrice = Math.random() * 25 + 15; // $15-40 range
  return Math.round(basePrice * 100) / 100; // Round to 2 decimal places
}

// Helper function to generate realistic ratings
function generateRandomRating() {
  const rating = Math.random() * 1.5 + 3.5; // 3.5-5.0 range
  return Math.round(rating * 10) / 10; // Round to 1 decimal place
}

// Helper function to generate descriptions based on title
function generateDescription(title) {
  const descriptions = {
    'pizza': 'Deliciosa pizza con ingredientes frescos',
    'pasta': 'Pasta casera con salsa tradicional',
    'salmon': 'Salmón fresco preparado a la perfección',
    'chicken': 'Pollo tierno y jugoso',
    'beef': 'Carne de res de la mejor calidad',
    'vegetarian': 'Opción vegetariana nutritiva y sabrosa',
    'soup': 'Sopa caliente y reconfortante',
    'salad': 'Ensalada fresca con ingredientes de temporada',
    'dessert': 'Postre dulce para terminar perfectamente'
  };
  
  const titleLower = title.toLowerCase();
  for (const [key, description] of Object.entries(descriptions)) {
    if (titleLower.includes(key)) {
      return description;
    }
  }
  
  return `Delicioso ${title} preparado con ingredientes frescos`;
}

// Helper function to get emoji based on product name
function getProductEmoji(productName) {
  const name = productName?.toLowerCase() || '';
  if (name.includes('pizza')) return '🍕';
  if (name.includes('burger') || name.includes('hamburguesa')) return '🍔';
  if (name.includes('sushi') || name.includes('salmon') || name.includes('pescado') || name.includes('fish')) return '🍣';
  if (name.includes('pasta') || name.includes('risotto') || name.includes('spaghetti')) return '🍝';
  if (name.includes('ensalada') || name.includes('salad')) return '🥗';
  if (name.includes('pollo') || name.includes('chicken')) return '🍗';
  if (name.includes('beef') || name.includes('carne')) return '🥩';
  if (name.includes('soup') || name.includes('sopa')) return '🍲';
  if (name.includes('dessert') || name.includes('postre') || name.includes('cake')) return '🍰';
  if (name.includes('bevida') || name.includes('drink') || name.includes('coca')) return '🥤';
  if (name.includes('taco')) return '🌮';
  if (name.includes('sandwich')) return '🥪';
  return '🍽️';
}
