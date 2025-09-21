"use client";
import { useEffect, useState } from "react";
import Sidebar from "../../components/Sidebar";

export default function Profile() {
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchProfile();
  }, []);

  const fetchProfile = async () => {
    try {
      const response = await fetch("/api/user/profile");
      const data = await response.json();
      setProfile(data);
    } catch (error) {
      console.error('Error fetching profile:', error);
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
            <p className="text-xl text-gray-600 font-medium">Cargando tu perfil...</p>
          </div>
        </div>
      </div>
    );
  }

  if (!profile) {
    return (
      <div className="flex min-h-screen">
        <Sidebar />
        <div className="flex-1 flex items-center justify-center">
          <p className="text-xl text-gray-600">Error al cargar el perfil</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <div className="flex-1 overflow-auto">
        <div className="min-h-screen bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50">
          <div className="p-8">
            <div className="max-w-4xl mx-auto">
              <div className="text-center mb-10">
                <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-4">
                  👤 Mi Perfil
                </h1>
                <p className="text-lg text-gray-600">Gestiona tu información personal y preferencias</p>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Profile Card */}
                <div className="lg:col-span-1">
                  <div className="bg-white/70 backdrop-blur-sm rounded-xl shadow-lg p-6 text-center">
                    <div className="w-32 h-32 bg-gradient-to-br from-blue-400 to-purple-400 rounded-full mx-auto mb-4 flex items-center justify-center">
                      <span className="text-4xl font-bold text-white">
                        {profile.name ? profile.name.charAt(0).toUpperCase() : '?'}
                      </span>
                    </div>
                    <h2 className="text-2xl font-bold text-gray-900 mb-2">{profile.name || 'Usuario'}</h2>
                    <div className="flex justify-center mb-4">
                      <span className="bg-gradient-to-r from-yellow-400 to-yellow-600 text-white px-3 py-1 rounded-full text-sm font-medium">
                        ⭐ {profile.membershipLevel || 'Básico'}
                      </span>
                    </div>
                    <div className="space-y-3">
                      <button 
                        key="edit-profile"
                        className="w-full bg-gradient-to-r from-blue-500 to-purple-500 text-white font-medium py-2 px-4 rounded-lg hover:from-blue-600 hover:to-purple-600 transition-all duration-200 transform hover:scale-105"
                      >
                        ✏️ Editar Perfil
                      </button>
                      <button 
                        key="change-password"
                        className="w-full border-2 border-purple-300 text-purple-600 font-medium py-2 px-4 rounded-lg hover:bg-purple-50 transition-all duration-200"
                      >
                        🔒 Cambiar Contraseña
                      </button>
                    </div>
                  </div>
                </div>

                {/* Information Cards */}
                <div className="lg:col-span-2 space-y-6">
                  {/* Contact Information */}
                  <div className="bg-white/70 backdrop-blur-sm rounded-xl shadow-lg p-6">
                    <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center">
                      📞 Información de Contacto
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <label className="text-sm font-medium text-gray-600">Email</label>
                        <p className="text-gray-900 font-medium">{profile.email || 'No especificado'}</p>
                      </div>
                      <div>
                        <label className="text-sm font-medium text-gray-600">Teléfono</label>
                        <p className="text-gray-900 font-medium">{profile.phone || 'No especificado'}</p>
                      </div>
                    </div>
                  </div>

                  {/* Address Information */}
                  <div className="bg-white/70 backdrop-blur-sm rounded-xl shadow-lg p-6">
                    <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center">
                      🏠 Dirección
                    </h3>
                    {profile.address ? (
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <label className="text-sm font-medium text-gray-600">Calle</label>
                          <p className="text-gray-900 font-medium">{profile.address.street}</p>
                        </div>
                        <div>
                          <label className="text-sm font-medium text-gray-600">Ciudad</label>
                          <p className="text-gray-900 font-medium">{profile.address.city}</p>
                        </div>
                        <div>
                          <label className="text-sm font-medium text-gray-600">País</label>
                          <p className="text-gray-900 font-medium">{profile.address.country}</p>
                        </div>
                        <div>
                          <label className="text-sm font-medium text-gray-600">Código Postal</label>
                          <p className="text-gray-900 font-medium">{profile.address.zipCode}</p>
                        </div>
                      </div>
                    ) : (
                      <p className="text-gray-600">No hay dirección registrada</p>
                    )}
                  </div>

                  {/* Account Statistics */}
                  <div className="bg-white/70 backdrop-blur-sm rounded-xl shadow-lg p-6">
                    <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center">
                      📊 Estadísticas de Cuenta
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div className="text-center p-4 bg-gradient-to-br from-green-100 to-emerald-100 rounded-lg">
                        <div className="text-2xl font-bold text-green-600">{profile.totalOrders || 0}</div>
                        <div className="text-sm text-green-700">Pedidos Totales</div>
                      </div>
                      <div className="text-center p-4 bg-gradient-to-br from-blue-100 to-cyan-100 rounded-lg">
                        <div className="text-2xl font-bold text-blue-600">${(profile.totalSpent || 0).toFixed(2)}</div>
                        <div className="text-sm text-blue-700">Total Gastado</div>
                      </div>
                      <div className="text-center p-4 bg-gradient-to-br from-purple-100 to-pink-100 rounded-lg">
                        <div className="text-2xl font-bold text-purple-600">
                          {profile.joinDate ? new Date(profile.joinDate).toLocaleDateString('es-ES', { year: 'numeric' }) : 'N/A'}
                        </div>
                        <div className="text-sm text-purple-700">Miembro desde</div>
                      </div>
                    </div>
                  </div>

                  {/* Preferences */}
                  {profile.preferences && profile.preferences.length > 0 && (
                    <div className="bg-white/70 backdrop-blur-sm rounded-xl shadow-lg p-6">
                      <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center">
                        ❤️ Preferencias
                      </h3>
                      <div className="flex flex-wrap gap-2">
                        {profile.preferences.map((preference, index) => (
                          <span 
                            key={`preference-${index}`}
                            className="bg-gradient-to-r from-pink-200 to-purple-200 text-purple-800 px-3 py-1 rounded-full text-sm font-medium"
                          >
                            {preference}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
