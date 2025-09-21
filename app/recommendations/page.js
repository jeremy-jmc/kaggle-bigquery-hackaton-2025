"use client";
import { useEffect, useState } from "react";
import Sidebar from "../../components/Sidebar";

export default function Recommendations() {
  const [recs, setRecs] = useState([]);
  const [loading, setLoading] = useState(true);

  const loadRecs = async () => {
    try {
      setLoading(true);
      const response = await fetch("/api/user/recommendations");
      const data = await response.json();
      setRecs(data);
    } catch (error) {
      console.error('Error fetching recommendations:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadRecs();
  }, []);

  if (loading) {
    return (
      <div className="flex min-h-screen">
        <Sidebar />
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <div className="w-16 h-16 border-4 border-cyan-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
            <p className="text-xl text-gray-600 font-medium">Generando recomendaciones con IA...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <div className="flex-1 overflow-auto">
        <div className="min-h-screen bg-gradient-to-br from-cyan-50 via-blue-50 to-purple-50">
          <div className="p-8">
            <div className="max-w-6xl mx-auto">
              <div className="text-center mb-10">
                <h1 className="text-4xl font-bold bg-gradient-to-r from-cyan-600 to-purple-600 bg-clip-text text-transparent mb-4">
                  Recomendaciones IA
                </h1>
                <p className="text-lg text-gray-600 max-w-2xl mx-auto">
                  Productos personalizados seleccionados por nuestra inteligencia artificial basándose en tus gustos y comportamiento de compra
                </p>
                <button 
                  className="mt-4 bg-gradient-to-r from-cyan-500 to-purple-500 text-white px-6 py-2 rounded-lg hover:from-cyan-600 hover:to-purple-600 transition-all duration-200 transform hover:scale-105"
                  onClick={loadRecs}
                >
                  🔄 Generar Nuevas Recomendaciones
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {recs.map((rec, index) => (
                  <div key={index} className="bg-white/70 backdrop-blur-sm rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 overflow-hidden group">
                    <div className="relative">
                      <div className="h-48 bg-gradient-to-br from-cyan-100 to-purple-100 overflow-hidden">
                        {rec.imageUrl ? (
                          <img 
                            src={rec.imageUrl}
                            alt={rec.product}
                            className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500"
                            onError={(e) => {
                              // If Cloud Storage image fails, try fallback or show placeholder
                              if (rec.fallbackImageUrl && e.target.src !== rec.fallbackImageUrl) {
                                e.target.src = rec.fallbackImageUrl;
                              } else {
                                // Show styled placeholder
                                e.target.style.display = 'none';
                                e.target.nextSibling.style.display = 'flex';
                              }
                            }}
                          />
                        ) : null}
                        {/* Fallback placeholder */}
                        <div className={`w-full h-full flex items-center justify-center ${rec.imageUrl ? 'hidden' : 'flex'}`}>
                          <div className="text-4xl">�️</div>
                        </div>
                      </div>
                      <div className="absolute top-3 right-3">
                        <span className="bg-gradient-to-r from-cyan-500 to-purple-500 text-white px-2 py-1 rounded-full text-xs font-medium">
                          {rec.confidence ? `${rec.confidence}% Match` : 'IA Match'}
                        </span>
                      </div>
                    </div>
                    
                    <div className="p-6">
                      <div className="flex items-start justify-between mb-2">
                        <h3 className="text-lg font-semibold text-gray-900 group-hover:text-purple-600 transition-colors flex-1">
                          {rec.emoji && <span className="mr-2">{rec.emoji}</span>}
                          {rec.product}
                        </h3>
                      </div>
                      
                      {/* Rating and Price */}
                      <div className="flex items-center justify-between mb-3">
                        {rec.rating && (
                          <div className="flex items-center space-x-1">
                            {[...Array(5)].map((_, i) => (
                              <span key={i} className={`text-sm ${i < Math.floor(rec.rating) ? 'text-yellow-400' : 'text-gray-300'}`}>
                                ⭐
                              </span>
                            ))}
                            <span className="text-xs text-gray-600 ml-1">({rec.rating})</span>
                          </div>
                        )}
                        {rec.price && (
                          <span className="text-lg font-bold text-green-600">
                            ${rec.price}
                          </span>
                        )}
                      </div>
                      
                      {/* Description if available */}
                      {rec.description && (
                        <p className="text-sm text-gray-600 mb-3 line-clamp-2">{rec.description}</p>
                      )}
                      
                      <div className="mb-4">
                        <p className="text-xs text-gray-500 mb-1">¿Por qué te recomendamos esto?</p>
                        <p className="text-sm text-purple-600 font-medium">{rec.reason}</p>
                      </div>
                      
                      <button className="w-full bg-gradient-to-r from-cyan-500 to-purple-500 text-white font-medium py-2 px-4 rounded-lg hover:from-cyan-600 hover:to-purple-600 transition-all duration-200 transform hover:scale-105">
                        Ver Producto
                      </button>
                    </div>
                  </div>
                ))}
              </div>
              
              {recs.length === 0 && (
                <div className="text-center py-16">
                  <div className="text-6xl mb-4">🤖</div>
                  <h3 className="text-2xl font-semibold text-gray-700 mb-2">Generando Recomendaciones</h3>
                  <p className="text-gray-600 max-w-md mx-auto">
                    Nuestra IA está analizando tus preferencias para ofrecerte las mejores recomendaciones personalizadas
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
