export async function GET() {
  try {
    // Check if required environment variables are set
    const projectId = process.env.GOOGLE_CLOUD_PROJECT_ID;
    const credentials = process.env.GOOGLE_APPLICATION_CREDENTIALS;
    const privateKey = process.env.GOOGLE_CLOUD_PRIVATE_KEY;
    const clientEmail = process.env.GOOGLE_CLOUD_CLIENT_EMAIL;

    console.log('Environment Variables:');
    console.log('GOOGLE_CLOUD_PROJECT_ID:', projectId ? 'Set' : 'Not Set');
    console.log('GOOGLE_APPLICATION_CREDENTIALS:', credentials ? 'Set' : 'Not Set');
    console.log('GOOGLE_CLOUD_PRIVATE_KEY:', privateKey ? 'Set' : 'Not Set');
    console.log('GOOGLE_CLOUD_CLIENT_EMAIL:', clientEmail ? 'Set' : 'Not Set');

    if (!projectId) {
      return Response.json({
        status: 'error',
        message: 'Missing GOOGLE_CLOUD_PROJECT_ID in environment variables',
        help: 'Add your Google Cloud Project ID to .env.local file'
      }, { status: 400 });
    }

    if (!credentials && (!privateKey || !clientEmail)) {
      return Response.json({
        status: 'error',
        message: 'Missing BigQuery credentials',
        help: 'Either set GOOGLE_APPLICATION_CREDENTIALS path or provide GOOGLE_CLOUD_PRIVATE_KEY and GOOGLE_CLOUD_CLIENT_EMAIL',
        environment: {
          hasProjectId: !!projectId,
          hasCredentialsFile: !!credentials,
          hasPrivateKey: !!privateKey,
          hasClientEmail: !!clientEmail
        }
      }, { status: 400 });
    }

    // Try to import and test BigQuery only if credentials are available
    const { testConnection, executeQuery } = await import('../../lib/bigquery');
    
    // Test basic connection
    const connectionTest = await testConnection();
    
    if (!connectionTest) {
      return Response.json({
        status: 'error',
        message: 'Failed to connect to BigQuery',
        details: 'Check your credentials and project configuration'
      }, { status: 500 });
    }

    // Test a simple query
    const testQuery = 'SELECT "BigQuery connection successful!" as message, CURRENT_TIMESTAMP() as timestamp';
    const result = await executeQuery(testQuery);

    return Response.json({
      status: 'success',
      message: 'BigQuery integration is working!',
      connection: connectionTest,
      testResult: result[0],
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('BigQuery test error:', error);
    
    return Response.json({
      status: 'error',
      message: 'BigQuery test failed',
      error: error.message,
      stack: error.stack,
      suggestions: [
        'Verify your GOOGLE_CLOUD_PROJECT_ID in .env.local',
        'Check your service account credentials',
        'Ensure BigQuery API is enabled in your Google Cloud project',
        'Verify your dataset and table names are correct'
      ]

    }, { status: 500 });
  }
}