"use client";
import { useEffect, useState } from "react";
import Sidebar from "../../components/Sidebar";

export default function Orders() {
  const [orders, setOrders] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchOrders();
  }, []);

  const fetchOrders = async () => {
    try {
      const response = await fetch("/api/user/orders");
      const data = await response.json();
      setOrders(data);
    } catch (error) {
      console.error('Error fetching orders:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex min-h-screen">
        <Sidebar />
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <div className="w-16 h-16 border-4 border-purple-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
            <p className="text-xl text-gray-600 font-medium">Cargando tus pedidos...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <div className="flex-1 p-8 overflow-auto">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-3xl font-bold text-gray-900 mb-8">Historial de Pedidos</h1>
          <div className="space-y-6">
            {orders.map((order, index) => (
              <div key={order.id} className="bg-white rounded-lg shadow-md p-6">
                <div className="flex justify-between items-start mb-4">
                  <div>
                    <h3 className="text-lg font-semibold">Pedido #{order.id}</h3>
                    <p className="text-gray-600">Fecha: {new Date(order.date).toLocaleDateString('es-ES')}</p>
                  </div>
                  <div className="text-right">
                    <span className="inline-block px-3 py-1 text-sm font-medium rounded-full bg-blue-100 text-blue-800">
                      {order.status || 'Pendiente'}
                    </span>
                    <p className="text-lg font-bold text-gray-900 mt-1">${(order.total || 0).toFixed(2)}</p>
                  </div>
                </div>
                
                <div className="border-t pt-4">
                  <h4 className="font-medium text-gray-700 mb-2">Productos:</h4>
                  <div className="space-y-2">
                    {(order.items || []).map((item, itemIndex) => (
                      <div key={`item-${order.id}-${itemIndex}`} className="flex justify-between">
                        <span>{item.name || 'Producto'} x{item.quantity || 1}</span>
                        <span>${(item.price || 0).toFixed(2)}</span>
                      </div>
                    ))}
                  </div>
                </div>
                
                <div className="mt-4 text-sm text-gray-600">
                  <strong>Dirección:</strong> {order.shippingAddress || 'No especificada'}
                </div>
              </div>
            ))}
          </div>
          
          {orders.length === 0 && (
            <div className="text-center py-12">
              <p className="text-gray-600 text-lg">No tienes pedidos aún</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
