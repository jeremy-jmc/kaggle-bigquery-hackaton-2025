'use client'

import { useState } from 'react';
import { usePathname } from 'next/navigation';
import Link from 'next/link';

export default function Sidebar() {
  const [isOpen, setIsOpen] = useState(true);
  const pathname = usePathname();

  const menuItems = [
    {
      href: '/',
      label: 'Catalog',
      icon: '🛍️',
      description: 'Explore amazing products',
      gradient: 'from-purple-500 to-pink-500'
    },
    {
      href: '/profile',
      label: 'My Profile',
      icon: '👤',
      description: 'Your personal information',
      gradient: 'from-blue-500 to-cyan-500'
    },
    {
      href: '/orders',
      label: 'My Recipes',
      icon: '📦',
      description: 'Recipes history and details',
      gradient: 'from-green-500 to-emerald-500'
    },
    {
      href: '/recommendations',
      label: 'Recommendations',
      icon: '🎯',
      description: 'Personalized suggestions',
      gradient: 'from-orange-500 to-red-500'
    }
  ];

  const isActive = (href) => {
    if (href === '/') {
      return pathname === '/';
    }
    return pathname.startsWith(href);
  };

  return (
    <div className={`bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white transition-all duration-500 ${isOpen ? 'w-80' : 'w-20'} min-h-screen flex flex-col shadow-2xl relative overflow-hidden`}>
      {/* Decorative background */}
      <div className="absolute inset-0 bg-gradient-to-br from-purple-600/20 to-pink-600/20 opacity-50"></div>
      <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-blue-400/30 to-purple-400/30 rounded-full blur-3xl"></div>
      <div className="absolute bottom-0 left-0 w-24 h-24 bg-gradient-to-br from-pink-400/30 to-red-400/30 rounded-full blur-2xl"></div>
      
      {/* Header */}
      <div className="relative z-10 p-6 border-b border-white/10 backdrop-blur-sm">
        <div className="flex items-center justify-between">
          {isOpen && (
            <div className="animate-slide-in">
              <h1 className="text-2xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                Mi Recipes Book ✨
              </h1>
              <p className="text-sm text-purple-200 mt-1">User panel</p>
            </div>
          )}
          <button
            onClick={() => setIsOpen(!isOpen)}
            className="p-3 bg-white/10 backdrop-blur-sm rounded-xl hover:bg-white/20 transition-all duration-300 hover:scale-110 border border-white/20"
            title={isOpen ? 'Contraer sidebar' : 'Expandir sidebar'}
          >
            <span className="text-lg">{isOpen ? '←' : '→'}</span>
          </button>
        </div>
      </div>

      {/* Navigation */}
      <nav className="relative z-10 flex-1 p-4">
        <ul className="space-y-3">
          {menuItems.map((item, index) => (
            <li key={item.href} className="animate-slide-in" style={{ animationDelay: `${index * 0.1}s` }}>
              <Link
                href={item.href}
                className={`group flex items-center p-4 rounded-2xl transition-all duration-300 relative overflow-hidden ${
                  isActive(item.href)
                    ? 'bg-gradient-to-r ' + item.gradient + ' shadow-lg scale-105 transform'
                    : 'text-purple-200 hover:bg-white/10 hover:text-white hover:scale-105 transform'
                }`}
                title={!isOpen ? item.label : ''}
              >
                {/* Icon with animated background */}
                <div className={`text-2xl mr-4 p-2 rounded-xl transition-all duration-300 ${
                  isActive(item.href) 
                    ? 'bg-white/20 backdrop-blur-sm' 
                    : 'group-hover:bg-white/10'
                }`}>
                  {item.icon}
                </div>
                
                {isOpen && (
                  <div className="flex-1">
                    <div className="font-semibold text-lg">{item.label}</div>
                    <div className={`text-sm transition-colors duration-300 ${
                      isActive(item.href) ? 'text-white/90' : 'text-purple-300 group-hover:text-purple-100'
                    }`}>
                      {item.description}
                    </div>
                  </div>
                )}
                
                {isOpen && isActive(item.href) && (
                  <div className="w-3 h-3 bg-white rounded-full shadow-lg animate-pulse"></div>
                )}
                
                {/* Hover effect */}
                <div className="absolute inset-0 bg-gradient-to-r from-white/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-2xl"></div>
              </Link>
            </li>
          ))}
        </ul>
      </nav>

      {/* User Info */}
      {isOpen && (
        <div className="relative z-10 p-6 border-t border-white/10 backdrop-blur-sm animate-slide-in">
          <div className="flex items-center space-x-4 p-4 bg-white/10 rounded-2xl backdrop-blur-sm border border-white/20">
            <div className="w-12 h-12 bg-gradient-to-br from-purple-400 to-pink-400 rounded-full flex items-center justify-center shadow-lg">
              <span className="text-lg font-bold text-white">JP</span>
            </div>
            <div className="flex-1">
              <p className="font-semibold text-white">Juan Pérez</p>
              <p className="text-sm text-purple-200">juan.perez@example.com</p>
            </div>
          </div>
          <button className="w-full mt-4 p-3 text-sm text-purple-200 hover:text-white hover:bg-white/10 rounded-xl transition-all duration-300 border border-white/10 hover:border-white/30">
            Close session 🚪
          </button>
        </div>
      )}
    </div>
  );
}
