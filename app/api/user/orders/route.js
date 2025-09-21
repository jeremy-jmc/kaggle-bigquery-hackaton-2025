export async function GET() {
  return Response.json([
    {
      id: "ORD-001",
      date: "2025-08-01T14:30:00.000Z",
      status: "Entregado",
      total: 28.99,
      shippingAddress: "Calle Mayor 123, Madrid, España",
      items: [
        { name: "Pizza Pepperoni", quantity: 1, price: 18.99 },
        { name: "Coca Cola 500ml", quantity: 2, price: 5.00 }
      ]
    },
    {
      id: "ORD-002",
      date: "2025-08-10T16:45:00.000Z",
      status: "En camino",
      total: 45.50,
      shippingAddress: "Avenida de la Paz 456, Barcelona, España",
      items: [
        { name: "Hamburguesa Doble", quantity: 2, price: 15.99 },
        { name: "Papas Fritas Grandes", quantity: 2, price: 6.50 },
        { name: "Milkshake Chocolate", quantity: 1, price: 7.50 }
      ]
    },
    {
      id: "ORD-003",
      date: "2025-09-15T12:20:00.000Z",
      status: "Preparando",
      total: 22.75,
      shippingAddress: "Plaza del Sol 789, Valencia, España",
      items: [
        { name: "Ensalada César", quantity: 1, price: 12.99 },
        { name: "Agua Mineral", quantity: 2, price: 2.50 },
        { name: "Pan de Ajo", quantity: 1, price: 4.75 }
      ]
    }
  ]);
}
