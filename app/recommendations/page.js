"use client";
import { useEffect, useState } from "react";
import Sidebar from "../../components/Sidebar";

export default function Recommendations() {
  const [recs, setRecs] = useState([]);
  const [loading, setLoading] = useState(true);

  // Format possibly array/JSON strings into human-readable
  const formatList = (v) => {
    if (!v) return null;
    if (Array.isArray(v)) return v.join(', ');
    try {
      const parsed = JSON.parse(v);
      if (Array.isArray(parsed)) return parsed.join(', ');
    } catch (_) {}
    return String(v);
  };

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
                  Recommendations
                </h1>
                <p className="text-lg text-gray-600 max-w-2xl mx-auto">
                  Products personalized by our artificial intelligence based on your tastes and shopping behavior
                </p>
                {recs && recs.length > 0 && recs[0].why && (
                  <div className="mt-6 text-left max-w-3xl mx-auto bg-white/70 backdrop-blur-sm rounded-lg border border-purple-100 p-4 shadow-sm">
                    <h2 className="text-sm font-semibold text-gray-700 mb-2">Your preference profile</h2>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-sm text-gray-700">
                      {recs[0].why.diet && (
                        <div>
                          <span className="text-gray-500">Diet: </span>
                          <span>{formatList(recs[0].why.diet)}</span>
                        </div>
                      )}
                      {recs[0].why.convenience && (
                        <div>
                          <span className="text-gray-500">Convenience: </span>
                          <span>{formatList(recs[0].why.convenience)}</span>
                        </div>
                      )}
                      {recs[0].why.flavorPrefs && (
                        <div>
                          <span className="text-gray-500">Flavor: </span>
                          <span>{formatList(recs[0].why.flavorPrefs)}</span>
                        </div>
                      )}
                      {recs[0].why.foodPrefs && (
                        <div>
                          <span className="text-gray-500">Food: </span>
                          <span>{formatList(recs[0].why.foodPrefs)}</span>
                        </div>
                      )}
                    </div>
                    <p className="text-xs text-gray-500 mt-2">Based on your historical interactions</p>
                  </div>
                )}
                <button 
                  className="mt-4 bg-gradient-to-r from-cyan-500 to-purple-500 text-white px-6 py-2 rounded-lg hover:from-cyan-600 hover:to-purple-600 transition-all duration-200 transform hover:scale-105"
                  onClick={loadRecs}
                >
                  🔄 Generate new recommendations
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
                        {rec.rating && (
                            <div className="flex items-center mb-2">
                              <div className="flex items-center">
                                {[...Array(5)].map((_, i) =>
                                  i < rec.rating ? (
                                    <span key={i} className="text-lg text-yellow-400">
                                      ⭐
                                    </span>
                                  ) : null
                                )}
                                <span className="text-sm text-gray-600 ml-2">({rec.rating}/5)</span>
                              </div>
                            </div>
                          )}
                      
                      {/* Description if available */}
                      {rec.description && (
                        <p className="text-sm text-gray-600 mb-3 line-clamp-2">{rec.description}</p>
                      )}
                      
                      {/* Per-card why block removed in favor of top summary */}

                    </div>
                  </div>
                ))}
              </div>
              
              {recs.length === 0 && (
                <div className="text-center py-16">
                  <div className="text-6xl mb-4">🤖</div>
                  <h3 className="text-2xl font-semibold text-gray-700 mb-2">Generating Recommendations</h3>
                  <p className="text-gray-600 max-w-md mx-auto">
                    Our AI is analyzing your preferences to provide you with the best personalized recommendations
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
