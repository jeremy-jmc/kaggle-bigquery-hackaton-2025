import { Storage } from '@google-cloud/storage';
import { NextResponse } from 'next/server';

let storageClient = null;

/**
 * Initialize Google Cloud Storage client
 */
function getStorageClient() {
  if (!storageClient) {
    const options = {
      projectId: process.env.GOOGLE_CLOUD_PROJECT_ID,
    };

    // Use the same authentication as BigQuery
    if (process.env.GOOGLE_APPLICATION_CREDENTIALS) {
      options.keyFilename = process.env.GOOGLE_APPLICATION_CREDENTIALS;
    } else if (process.env.GOOGLE_CLOUD_PRIVATE_KEY && process.env.GOOGLE_CLOUD_CLIENT_EMAIL) {
      options.credentials = {
        private_key: process.env.GOOGLE_CLOUD_PRIVATE_KEY.replace(/\\n/g, '\n'),
        client_email: process.env.GOOGLE_CLOUD_CLIENT_EMAIL,
      };
    }

    storageClient = new Storage(options);
  }
  return storageClient;
}

export async function GET(request, { params }) {
  try {
    const { recipeId } = params;
    
    if (!recipeId) {
      return new NextResponse('Recipe ID is required', { status: 400 });
    }

    const storage = getStorageClient();
    const bucketName = 'kaggle-recipes';
    
    // Try different possible file paths/extensions
    const possiblePaths = [
      `core-data-images/core-data-images/${recipeId}`,
      `core-data-images/${recipeId}`,
      `${recipeId}`,
      `${recipeId}.jpg`,
      `${recipeId}.png`,
      `${recipeId}.jpeg`,
      `core-data-images/core-data-images/${recipeId}.jpg`,
      `core-data-images/core-data-images/${recipeId}.png`,
      `core-data-images/core-data-images/${recipeId}.jpeg`
    ];

    for (const filePath of possiblePaths) {
      try {
        const file = storage.bucket(bucketName).file(filePath);
        
        // Check if file exists
        const [exists] = await file.exists();
        if (!exists) continue;

        // Get file metadata to determine content type
        const [metadata] = await file.getMetadata();
        const contentType = metadata.contentType || 'image/jpeg';

        // Download the file
        const [contents] = await file.download();

        // Return the image with proper headers
        return new NextResponse(contents, {
          status: 200,
          headers: {
            'Content-Type': contentType,
            'Cache-Control': 'public, max-age=86400', // Cache for 24 hours
            'Access-Control-Allow-Origin': '*',
          },
        });
      } catch (error) {
        console.log(`Failed to fetch ${filePath}:`, error.message);
        continue;
      }
    }

    // If no image found, return a 404
    return new NextResponse('Image not found', { status: 404 });

  } catch (error) {
    console.error('Error fetching image from Cloud Storage:', error);
    return new NextResponse('Internal server error', { status: 500 });
  }
}