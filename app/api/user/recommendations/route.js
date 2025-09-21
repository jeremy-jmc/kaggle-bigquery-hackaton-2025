export async function GET() {
  const recs = [
    { product: "Risotto de mariscos", reason: "Similar a tus preferencias de pescados y pastas." },
    { product: "Salmón a la plancha", reason: "Basado en tu consumo frecuente de sushi." }
  ];
  return Response.json(recs);
}
