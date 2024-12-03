import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    Tooltip,
    Legend,
    PieChart,
    Pie,
    Cell,
    ResponsiveContainer
} from 'recharts';

// Custom Table Components
const Table = ({ children }) => (
    <div className="w-full overflow-x-auto">
        <table className="w-full border-collapse bg-white">{children}</table>
    </div>
);

const TableHeader = ({ children }) => (
    <thead className="bg-gray-100 border-b">
        {children}
    </thead>
);

const TableRow = ({ children, className = '' }) => (
    <tr className={`border-b hover:bg-gray-50 ${className}`}>{children}</tr>
);

const TableHead = ({ children }) => (
    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
        {children}
    </th>
);

const TableBody = ({ children }) => (
    <tbody className="divide-y divide-gray-200">{children}</tbody>
);

const TableCell = ({ children, className = '' }) => (
    <td className={`px-4 py-3 text-sm ${className}`}>{children}</td>
);

// Custom Badge Component
const Badge = ({ children, variant = 'default' }) => {
    const variantStyles = {
        default: 'bg-gray-200 text-gray-800',
        destructive: 'bg-red-200 text-red-800',
        warning: 'bg-yellow-200 text-yellow-800'
    };

    return (
        <span className={`px-2 py-1 rounded-full text-xs font-semibold ${variantStyles[variant]}`}>
            {children}
        </span>
    );
};

const HomeApp = () => {
    const [processedData, setProcessedData] = useState([]);
    const [etiquetaStats, setEtiquetaStats] = useState([]);
    const [categoryDistribution, setCategoryDistribution] = useState([]);
    const [isProcessingKeyAvailable, setIsProcessingKeyAvailable] = useState(false);

    useEffect(() => {
        const processingKey = localStorage.getItem('processingKey');

        if (processingKey) {
            setIsProcessingKeyAvailable(true);
            fetchProcessedData(processingKey);
        }
    }, []);

    const fetchProcessedData = async (processingKey) => {
        try {
            // Realizar la solicitud GET con el processingKey en la URL
            const response = await axios.get(`http://localhost:8000/processed-data/${processingKey}`);
    
            const mockApiData = response.data;

            console.log(response);
    
            if (Array.isArray(mockApiData)) {
                // Procesar estadísticas de etiquetas
                const etiquetaCounts = mockApiData.reduce((acc, item) => {
                    acc[item.etiqueta] = (acc[item.etiqueta] || 0) + 1;
                    return acc;
                }, {});
    
                const etiquetaStatsData = Object.entries(etiquetaCounts).map(([etiqueta, count]) => ({
                    etiqueta,
                    count
                }));
    
                // Procesar distribución por categoría
                const categoryCounts = mockApiData.reduce((acc, item) => {
                    const category = item.categoryId.slice(0, 2);
                    acc[category] = (acc[category] || 0) + 1;
                    return acc;
                }, {});
    
                const categoryData = Object.entries(categoryCounts).map(([category, count]) => ({
                    name: category,
                    value: count
                }));
    
                setEtiquetaStats(etiquetaStatsData);
                setCategoryDistribution(categoryData);
                setProcessedData(mockApiData);
            } else {
                console.error("La respuesta de la API no es un array:", mockApiData);
            }
        } catch (error) {
            console.error('Error fetching processed data:', error);
        }
    };    

    // Paleta de colores para gráficos
    const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

    return (
        <div className="p-8 bg-gray-50 min-h-screen">
            <h1 className="text-3xl font-bold mb-6">Análisis de Datos</h1>

            {!isProcessingKeyAvailable ? (
                <div className="text-center text-xl text-gray-600">
                    No hay datos procesados. Por favor, cargue un archivo.
                </div>
            ) : (
                <div className="grid grid-cols-3 gap-6">
                    <div className="bg-white p-6 rounded-lg shadow-md col-span-2">
                        <h3 className="text-xl font-semibold mb-4">Productos por Etiqueta</h3>
                        <ResponsiveContainer width="100%" height={300}>
                            <BarChart data={etiquetaStats}>
                                <XAxis dataKey="etiqueta" />
                                <YAxis />
                                <Tooltip />
                                <Legend />
                                <Bar dataKey="count" fill="#8884d8" />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                    <div className="bg-white p-6 rounded-lg shadow-md">
                        <h3 className="text-xl font-semibold mb-4">Productos por Categoría</h3>
                        <ResponsiveContainer width="100%" height={300}>
                            <PieChart>
                                <Pie
                                    data={categoryDistribution}
                                    cx="50%"
                                    cy="50%"
                                    labelLine={false}
                                    outerRadius={80}
                                    fill="#8884d8"
                                    dataKey="value"
                                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                                >
                                    {categoryDistribution.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                    ))}
                                </Pie>
                                <Tooltip />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                    <div className="bg-white p-6 rounded-lg shadow-md col-span-3">
                        <h3 className="text-xl font-semibold mb-4">Resumen de Productos</h3>
                        <Table>
                            <TableHeader>
                                <TableRow>
                                    <TableHead>Descripción</TableHead>
                                    <TableHead>Stock Actual</TableHead>
                                    <TableHead>Categoría</TableHead>
                                    <TableHead>Etiqueta</TableHead>
                                    <TableHead>Observación</TableHead>
                                </TableRow>
                            </TableHeader>
                            <TableBody>
                                {processedData.map((item, index) => (
                                    <TableRow key={index}>
                                        <TableCell>{item.productDescription}</TableCell>
                                        <TableCell>{item.stockDiaActual}</TableCell>
                                        <TableCell>{item.categoryId}</TableCell>
                                        <TableCell>
                                            <Badge
                                                variant={
                                                    item.etiqueta === 'Crítico' ? 'destructive' :
                                                        item.etiqueta === 'Alto' ? 'warning' : 'default'
                                                }
                                            >
                                                {item.etiqueta}
                                            </Badge>
                                        </TableCell>
                                        <TableCell>{item.observacion}</TableCell>
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    </div>
                </div>
            )}
        </div>
    );
};

export default HomeApp;
