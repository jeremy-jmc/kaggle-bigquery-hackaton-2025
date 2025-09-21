export async function GET() {
  return Response.json({
    id: 1,
    name: "Carlos Pérez",
    email: "carlos.perez@example.com",
    phone: "+34 666 123 456",
    age: 34,
    address: {
      street: "Calle Mayor 123",
      city: "Madrid",
      country: "España",
      zipCode: "28001"
    },
    preferences: ["Pescado", "Pastas"],
    joinDate: "2023-01-15T10:00:00.000Z",
    totalOrders: 15,
    totalSpent: 1234.56,
    avatar: "https://via.placeholder.com/150",
    membershipLevel: "Gold"
  });
}
