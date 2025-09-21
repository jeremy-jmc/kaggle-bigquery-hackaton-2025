'use client'

import { useState, useEffect } from 'react';
import Sidebar from '../components/Sidebar';

export default function Home() {
  const [products, setProducts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [currentSearch, setCurrentSearch] = useState(''); // Track active search
  const [selectedCategory, setSelectedCategory] = useState('All');

  useEffect(() => {
    fetchProducts();
  }, []);

  const fetchProducts = async (searchQuery = '') => {
    try {
      setLoading(true);
      let url = '/api/products';
      if (searchQuery && searchQuery.trim() !== '') {
        url += `?search=${encodeURIComponent(searchQuery.trim())}`;
      }
      
      const response = await fetch(url);
      const data = await response.json();
      setProducts(data);
      setCurrentSearch(searchQuery);
    } catch (error) {
      console.error('Error fetching products:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSearchKeyPress = (e) => {
    if (e.key === 'Enter') {
      fetchProducts(searchTerm);
    }
  };

  const handleSearchClear = () => {
    setSearchTerm('');
    setCurrentSearch('');
    fetchProducts(''); // Fetch default products
  };

  const categories = ['All', ...new Set(products.map(p => p.category))];
  
  // Always apply category filtering to retrieved items (both search results and default products)
  const filteredProducts = products.filter(product => {
    const matchesCategory = selectedCategory === 'All' || product.category === selectedCategory;
    return matchesCategory;
  });

  if (loading) {
    return (
      <div className="flex min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
        <Sidebar />
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <div className="w-16 h-16 border-4 border-purple-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
            <p className="text-xl text-gray-600 font-medium">Loading amazing products...</p>
          </div>
        </div>
      </div>
    );
  }


  return (
    <div className="flex min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      <Sidebar />
      
      <main className="flex-1 p-8 overflow-auto">
        <div className="max-w-7xl mx-auto">
          {/* Header with gradient */}
          <div className="mb-8 text-center">
            <h1 className="text-5xl font-bold bg-gradient-to-r from-purple-600 via-pink-600 to-blue-600 bg-clip-text text-transparent mb-4 animate-slide-in">
              ✨ Products Catalog ✨
            </h1>
            <p className="text-xl text-gray-600 animate-fade-in">
              {currentSearch 
                ? `Results for: "${currentSearch}"${selectedCategory !== 'Todos' ? ` • Category: ${selectedCategory}` : ''}`
                : selectedCategory !== 'Todos' 
                  ? `Showing products from: ${selectedCategory}`
                  : 'Discover amazing products at the best prices'
              }
            </p>
          </div>

          {/* Search and filters */}
          <div className="mb-8 bg-white/70 backdrop-blur-lg rounded-3xl p-6 shadow-xl border border-white/20 animate-slide-in">
            <div className="flex flex-col md:flex-row gap-4">
              <div className="flex-1">
                <div className="relative">
                  <input
                    type="text"
                    placeholder="🔍 Search items (press Enter)"
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    onKeyPress={handleSearchKeyPress}
                    className="w-full p-4 pl-12 pr-12 rounded-2xl border-2 border-purple-200 focus:border-purple-500 focus:outline-none transition-all duration-300 text-gray-700 bg-white/80"
                  />
                  <div className="absolute left-4 top-1/2 transform -translate-y-1/2 text-purple-400 text-xl">
                    🔍
                  </div>
                  {(searchTerm || currentSearch) && (
                    <button
                      onClick={handleSearchClear}
                      className="absolute right-4 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600 text-xl"
                    >
                      ✕
                    </button>
                  )}
                </div>
              </div>
              <div className="flex gap-2 flex-wrap">
                {categories.map(category => (
                  <button
                    key={category}
                    onClick={() => setSelectedCategory(category)}
                    className={`px-6 py-3 rounded-2xl font-medium transition-all duration-300 transform hover:scale-105 ${
                      selectedCategory === category
                        ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white shadow-lg'
                        : 'bg-white/80 text-gray-700 hover:bg-purple-100 border border-purple-200'
                    }`}
                  >
                    {category}
                  </button>
                ))}
              </div>
            </div>
          </div>
          
          {/* Products grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-8">
            {filteredProducts.map((product, index) => (
              <div 
                key={product.id} 
                className="group bg-white/80 backdrop-blur-lg rounded-3xl shadow-xl overflow-hidden hover:shadow-2xl transition-all duration-500 transform hover:scale-105 border border-white/20 animate-slide-in"
                style={{ animationDelay: `${index * 0.1}s` }}
              >
                {/* Product image with overlay */}
                <div className="relative h-56 bg-gradient-to-br from-purple-200 via-pink-200 to-blue-200 overflow-hidden">
                  {product.imageUrl ? (
                    <img 
                      src={product.imageUrl}
                      alt={product.name}
                      className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500"
                      onError={(e) => {
                        // If Cloud Storage image fails, try fallback or show placeholder
                        if (product.fallbackImageUrl && e.target.src !== product.fallbackImageUrl) {
                          e.target.src = product.fallbackImageUrl;
                        } else {
                          // Show styled placeholder
                          e.target.style.display = 'none';
                          e.target.nextSibling.style.display = 'flex';
                        }
                      }}
                    />
                  ) : null}
                  {/* Fallback placeholder */}
                  <div className={`absolute inset-0 flex items-center justify-center ${product.imageUrl ? 'hidden' : 'flex'}`}>
                    <div className="text-6xl opacity-30">{product.emoji || (product.category === 'Carnes' ? '🥩' : product.category === 'Ensaladas' ? '🥗' : product.category === 'Pasta' ? '🍝' : product.category === 'Sopas' ? '�' : product.category === 'Postres' ? '�' : product.category === 'Desayunos' ? '🥞' : '🍽️')}</div>
                  </div>
                  {/* Hover overlay */}
                  <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                </div>
                
                <div className="p-6">
                  {/* Category badge */}
                  <div className="mb-3">
                    <span className="px-3 py-1 text-xs bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-full font-medium">
                      {product.category}
                    </span>
                  </div>

                  <h3 className="text-xl font-bold text-gray-800 mb-2 group-hover:text-purple-600 transition-colors duration-300">
                    {product.name}
                  </h3>
                  <p className="text-sm text-gray-600 mb-4 line-clamp-2">{product.description}</p>
                  
                  {/* Rating */}
                  {product.rating && (
                    <div className="flex items-center mb-2">
                      <div className="flex items-center">
                        {[...Array(5)].map((_, i) =>
                          i < product.rating ? (
                            <span key={i} className="text-lg text-yellow-400">
                              ⭐
                            </span>
                          ) : null
                        )}
                        <span className="text-sm text-gray-600 ml-2">({product.rating}/5)</span>
                            </div>
                          </div>
                        )}
                  
                  
                  {/* Action buttons */}
                  <div className="space-y-3">
                    <button 
                      key={`cart-${product.id}`}
                      className={`w-full py-3 px-6 rounded-2xl font-bold transition-all duration-300 transform hover:scale-105 ${
                        product.inStock
                          ? 'bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white shadow-lg hover:shadow-xl'
                          : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                      }`}
                      disabled={!product.inStock}
                    >
                      👁️ See details
                    </button>
                    
                  </div>
                </div>
              </div>
            ))}
          </div>

          {filteredProducts.length === 0 && (
            <div className="text-center py-16">
              <div className="text-8xl mb-6">🔍</div>
              <h3 className="text-2xl font-bold text-gray-700 mb-4">Could not find any products</h3>
              <p className="text-gray-600 text-lg">Try different search terms or categories</p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
