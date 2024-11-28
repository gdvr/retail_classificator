import React from 'react';
import 'react-toastify/dist/ReactToastify.css';

// Componente Home
const HomeApp = () => (
    <div className="p-8 bg-gray-50 min-h-screen">
        <h1 className="text-3xl font-bold mb-6">Dashboard Principal</h1>
        <div className="grid grid-cols-3 gap-6">
            <div className="bg-white p-6 rounded-lg shadow-md">
                <h3 className="text-xl font-semibold mb-4">Estadísticas</h3>
            </div>
            <div className="bg-white p-6 rounded-lg shadow-md">
                <h3 className="text-xl font-semibold mb-4">Gráficos</h3>
            </div>
            <div className="bg-white p-6 rounded-lg shadow-md">
                <h3 className="text-xl font-semibold mb-4">Resumen</h3>
            </div>
        </div>
    </div>
);

export default HomeApp;