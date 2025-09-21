export async function GET() {
  const products = [
    { id: 1, name: "Pizza Pepperoni", price: 25, img: "/pizza.jpg" },
    { id: 2, name: "Hamburguesa Doble", price: 18, img: "/burger.jpg" },
    { id: 3, name: "Sushi Variado", price: 40, img: "/sushi.jpg" }
  ];
  return Response.json(products);
}
