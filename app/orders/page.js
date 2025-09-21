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
                    <h3 className="text-lg font-semibold text-gray-900">Pedido #{order.id}</h3>
                    <p className="text-gray-600">Fecha: {new Date(order.date).toLocaleDateString('es-ES')}</p>
                  </div>
                  <div className="text-right">
                    <span className="inline-block px-3 py-1 text-sm font-medium rounded-full bg-blue-100 text-blue-800">
                      {order.status || 'Pendiente'}
                    </span>
                  </div>
                </div>
                
                <div className="border-t pt-4">
                  <h4 className="font-medium text-gray-700 mb-2">Productos:</h4>
                  <div className="space-y-3">
                    {(order.items || []).map((item, itemIndex) => (
                      <div key={`item-${order.id}-${itemIndex}`} className="bg-gray-50 rounded-lg p-4">
                        <div className="flex gap-4">
                          {/* Recipe Image */}
                          <div className="flex-shrink-0 w-20 h-20 bg-gradient-to-br from-orange-100 to-red-100 rounded-lg overflow-hidden">
                            {item.imageUrl ? (
                              <img 
                                src={item.imageUrl}
                                alt={item.name}
                                className="w-full h-full object-cover"
                                onError={(e) => {
                                  // If Cloud Storage image fails, try fallback or show placeholder
                                  if (item.fallbackImageUrl && e.target.src !== item.fallbackImageUrl) {
                                    e.target.src = item.fallbackImageUrl;
                                  } else {
                                    // Show styled placeholder
                                    e.target.style.display = 'none';
                                    e.target.nextSibling.style.display = 'flex';
                                  }
                                }}
                              />
                            ) : null}
                            {/* Fallback placeholder */}
                            <div className={`w-full h-full flex items-center justify-center ${item.imageUrl ? 'hidden' : 'flex'}`}>
                              <div className="text-2xl">🍽️</div>
                            </div>
                          </div>
                          
                          {/* Item Details */}
                          <div className="flex-1">
                            <div className="flex justify-between items-start mb-2">
                              <span className="font-medium text-gray-800">{item.name || 'Producto'}</span>
                            </div>
                        
                        {/* Rating display */}
                        {item.rating && (
                          <div className="flex items-center mb-2">
                            <span className="text-sm text-gray-600 mr-2">Tu calificación:</span>
                            <div className="flex items-center">
                              {[...Array(5)].map((_, i) => (
                                <span key={i} className={`text-lg ${i < Math.floor(item.rating) ? 'text-yellow-400' : 'text-gray-300'}`}>
                                  ⭐
                                </span>
                              ))}
                              <span className="text-sm text-gray-600 ml-2">({item.rating}/5)</span>
                            </div>
                          </div>
                        )}
                        
                        {/* Comment display */}
                        {item.comment && (
                          <div className="mt-2">
                            <span className="text-sm text-gray-600">Tu comentario: </span>
                            <span className="text-sm text-gray-800 italic">"{item.comment}"</span>
                          </div>
                        )}
                        
                        {/* Recipe ID (optional, for debugging) */}
                        {item.recipeId && (
                          <div className="mt-1 text-xs text-gray-400">
                            Recipe ID: {item.recipeId}
                          </div>
                        )}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
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
