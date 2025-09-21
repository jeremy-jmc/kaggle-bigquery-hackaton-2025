export async function GET() {
  return Response.json({
    status: 'success',
    message: 'Test route is working!',
    timestamp: new Date().toISOString(),
    environment: {
      hasProjectId: !!process.env.GOOGLE_CLOUD_PROJECT_ID,
      hasCredentials: !!process.env.GOOGLE_APPLICATION_CREDENTIALS,
      hasPrivateKey: !!process.env.GOOGLE_CLOUD_PRIVATE_KEY,
      hasClientEmail: !!process.env.GOOGLE_CLOUD_CLIENT_EMAIL,
      projectId: process.env.GOOGLE_CLOUD_PROJECT_ID || 'Not set',
      datasetId: process.env.BIGQUERY_DATASET_ID || 'Not set'
    }
  });
}