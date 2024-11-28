import React from 'react';
import {
  Home,
  Files,
  Settings,
  BarChart2,
  CreditCard,
  Table
} from 'lucide-react';
import {
  Link,
  useLocation
} from 'react-router-dom';
import 'react-toastify/dist/ReactToastify.css';

// Componente de Sidebar
const Sidebar = ({ isOpen, onToggle }) => {
  const location = useLocation();
  const menuItems = [
    { 
      icon: <Home />, 
      label: 'Dashboard', 
      path: '/',
      active: location.pathname === '/'
    },
    { 
      icon: <Files />, 
      label: 'Subir Archivo', 
      path: '/archivos',
      active: location.pathname === '/archivos'
    },
    // { 
    //   icon: <BarChart2 />, 
    //   label: 'Analytics',
    //   path: '/analytics',
    //   active: location.pathname === '/analytics'
    // },
    { 
      icon: <Table />, 
      label: 'Archivo Cargados',
      path: '/archivos-cargados',
      active: location.pathname === '/archivos-cargados'
    }
  ];

  return (
    <div className={`
      fixed left-0 top-0 h-full 
      bg-white shadow-lg 
      transition-all duration-300
      ${isOpen ? 'w-64' : 'w-20'}
      overflow-hidden
    `}>
      <div className="p-4 flex items-center justify-between">
        <h2 className={`
          font-bold text-xl 
          transition-opacity duration-300
          ${isOpen ? 'opacity-100' : 'opacity-0'}
        `}>
          Mi Dashboard
        </h2>
        <button 
          onClick={onToggle} 
          className="p-2 hover:bg-gray-100 rounded-full"
        >
          {isOpen ? '←' : '→'}
        </button>
      </div>

      <nav className="mt-8">
        {menuItems.map((item, index) => (
          <Link 
            to={item.path} 
            key={index} 
            className={`
              flex items-center p-3 
              hover:bg-gray-100 
              cursor-pointer 
              no-underline text-black
              ${item.active ? 'bg-blue-50 border-r-4 border-blue-500' : ''}
            `}
          >
            <span className="mx-4">{item.icon}</span>
            <span 
              className={`
                transition-opacity duration-300
                ${isOpen ? 'opacity-100' : 'opacity-0'}
              `}
            >
              {item.label}
            </span>
          </Link>
        ))}
      </nav>
    </div>
  );
};

export default Sidebar;