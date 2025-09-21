import { executeQuery, CommonQueries } from '../../../lib/bigquery';

export async function GET(request) {
  try {
    // Extract user ID from query parameters or use default
    const { searchParams } = new URL(request.url);
    const userId = searchParams.get('userId') || '4368963';
    console.log('Fetching profile for userId:', userId);

    // Execute BigQuery to get user profile
    const profiles = await executeQuery(
      CommonQueries.getUserProfile(userId),
      {
        params: { userId: userId }
      }
    );

    if (profiles.length === 0) {
      return Response.json({ error: 'User not found' }, { status: 404 });
    }

    const profile = profiles[0];
    
    // Helper function to filter out None/null values from arrays
    const filterNoneValues = (arr) => {
      if (!arr || !Array.isArray(arr)) return [];
      return arr.filter(item => item && item !== 'None' && item !== null && item !== undefined);
    };

    // Build preferences array from ai_result fields, excluding None values
    const preferences = [];
    
    // Add cuisine preferences if they exist and are not None
    const cuisinePrefs = filterNoneValues(profile.cuisine_preferences);
    if (cuisinePrefs.length > 0) {
      preferences.push(...cuisinePrefs.map(pref => `Cocina: ${pref}`));
    }
    
    // Add daypart preferences if they exist and are not None
    const daypartPrefs = filterNoneValues(profile.daypart_preferences);
    if (daypartPrefs.length > 0) {
      preferences.push(...daypartPrefs.map(pref => `Horario: ${pref}`));
    }
    
    // Add dietary preferences if they exist and are not None
    const dietaryPrefs = filterNoneValues(profile.dietary_preferences);
    if (dietaryPrefs.length > 0) {
      preferences.push(...dietaryPrefs.map(pref => `Dieta: ${pref}`));
    }
    
    // Add food preferences if they exist and are not None
    const foodPrefs = filterNoneValues(profile.food_preferences);
    if (foodPrefs.length > 0) {
      preferences.push(...foodPrefs.map(pref => `Comida: ${pref}`));
    }
    
    // Transform data to match frontend expectations
    const transformedProfile = {
      id: profile.user_id,
      name: `Usuario ${profile.user_id}`,
      email: `user${profile.user_id}@foodrecsys.com`,
      phone: "+1 (555) 123-4567",
      age: 28,
      address: {
        street: "123 Food Street",
        city: "Culinary City",
        country: "Estados Unidos",
        zipCode: "12345"
      },
      preferences: preferences,
      joinDate: "2023-01-15T10:00:00.000Z",
      totalOrders: profile.n_history || 0,
      totalSpent: (profile.n_history || 0) * 25.50, // Estimated based on order history
      avatar: `https://ui-avatars.com/api/?name=Usuario+${profile.user_id}&background=6366f1&color=fff&size=150`,
      membershipLevel: getMembershipLevel((profile.n_history || 0) * 25.50),
      
      // Additional AI result data
      aiProfile: {
        userStory: profile.user_story,
        justification: profile.justification,
        diversityOpenness: profile.diversity_openness,
        conveniencePreference: profile.convenience_preference,
        dietaryPreference: profile.dietary_preference,
        cuisinePreference: profile.cuisine_preference,
        futurePreferences: profile.future_preferences,
        notes: profile.notes,
        status: profile.status,
        lifestyleTags: filterNoneValues(profile.lifestyle_tags),
        likedCuisines: filterNoneValues(profile.liked_cuisines),
        flavorPreferences: filterNoneValues(profile.flavor_preferences)
      }
    };

    return Response.json(transformedProfile);
  } catch (error) {
    console.error('Error fetching user profile:', error);
    
    // Fallback to mock data if BigQuery fails
    const fallbackProfile = {
      id: "1",
      name: "Usuario 1",
      email: "user1@foodrecsys.com",
      phone: "+1 (555) 123-4567",
      age: 28,
      address: {
        street: "123 Food Street",
        city: "Culinary City",
        country: "Estados Unidos",
        zipCode: "12345"
      },
      preferences: ["Cocina: Italiana", "Dieta: Vegetariana", "Horario: Cena"],
      joinDate: "2023-01-15T10:00:00.000Z",
      totalOrders: 15,
      totalSpent: 382.50,
      avatar: "https://ui-avatars.com/api/?name=Usuario+1&background=6366f1&color=fff&size=150",
      membershipLevel: "Gold"
    };
    
    return Response.json(fallbackProfile);
  }
}

// Helper functions
function getMembershipLevel(totalSpent) {
  if (totalSpent >= 500) return 'Platinum';
  if (totalSpent >= 200) return 'Gold';
  if (totalSpent >= 50) return 'Silver';
  return 'Básico';
}
