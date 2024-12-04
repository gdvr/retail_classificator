import axios from "axios";
import React, { useState } from "react";
import {
  Bar,
  BarChart,
  Legend,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

// Custom Table Components
const Table = ({ children }) => (
  <div className="w-full overflow-x-auto">
    <table className="w-full border-collapse bg-white">{children}</table>
  </div>
);

const TableHeader = ({ children }) => (
  <thead className="bg-gray-100 border-b">{children}</thead>
);

const TableRow = ({ children, className = "" }) => (
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

const TableCell = ({ children, className = "" }) => (
  <td className={`px-4 py-3 text-sm ${className}`}>{children}</td>
);

// Custom Badge Component
const Badge = ({ children, variant = "default" }) => {
  const variantStyles = {
    default: "bg-gray-200 text-gray-800",
    destructive: "bg-red-200 text-red-800",
    warning: "bg-yellow-200 text-yellow-800",
  };

  return (
    <span
      className={`px-2 py-1 rounded-full text-xs font-semibold ${variantStyles[variant]}`}
    >
      {children}
    </span>
  );
};

const HomeApp = () => {
  const [fecha, setFecha] = useState("");
  const [resultsData, setResultsData] = useState(null);
  const [distributionData, setDistributionData] = useState([]);

  const handleDateSubmit = async (e) => {
    e.preventDefault();

    try {
      // Call the new endpoint with the selected date
      const response = await axios.get(
        `http://localhost:5000/results?fecha=${fecha}T00:00:00`
      );

      // Process the counts from the response
      const countsData = response.data.counts;
      print(countsData);

      // Transform counts into chart-friendly format
      const distributionChartData = Object.entries(countsData)
        .filter(([key, value]) => value > 0)
        .map(([key, value]) => ({
          category: key,
          count: value,
        }));

      setResultsData(countsData);
      setDistributionData(distributionChartData);
    } catch (error) {
      console.error("Error fetching results:", error);
      // Optionally, set an error state to show to the user
    }
  };

  // Función para calcular el porcentaje total
  const calculatePercentage = () => {
    if (!resultsData) return 0;
    const total = Object.values(resultsData).reduce(
      (sum, count) => sum + count,
      0
    );
    return total;
  };

  return (
    <div className="p-8 bg-gray-50 min-h-screen">
      <h1 className="text-3xl font-bold mb-6">Análisis de Resultados</h1>

      {/* Date Input Form */}
      <form
        onSubmit={handleDateSubmit}
        className="mb-6 flex items-center space-x-4"
      >
        <label htmlFor="fecha" className="text-sm font-medium">
          Seleccionar Fecha:
        </label>
        <input
          type="date"
          id="fecha"
          value={fecha}
          onChange={(e) => setFecha(e.target.value)}
          className="border rounded px-3 py-2"
          required
        />
        <button
          type="submit"
          className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 transition"
        >
          Buscar Resultados
        </button>
      </form>

      {resultsData && (
        <div className="grid grid-cols-3 gap-6">
          <div className="bg-white p-6 rounded-lg shadow-md col-span-2">
            <h3 className="text-xl font-semibold mb-4">
              Distribución de Resultados (Barras)
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={distributionData}>
                <XAxis dataKey="category" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="count" fill="#8884d8" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-xl font-semibold mb-4">
              Análisis Detallado (Radar)
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart outerRadius="80%" data={distributionData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="category" />
                <PolarRadiusAxis
                  angle={30}
                  domain={[
                    0,
                    Math.max(...distributionData.map((d) => d.count)),
                  ]}
                />
                <Radar
                  name="Resultados"
                  dataKey="count"
                  stroke="#8884d8"
                  fill="#8884d8"
                  fillOpacity={0.6}
                />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md col-span-3">
            <h3 className="text-xl font-semibold mb-4">
              Resumen de Resultados
            </h3>
            <div className="mb-4">
              <p className="text-lg font-medium">
                Total de Resultados: {calculatePercentage()}
              </p>
            </div>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Categoría</TableHead>
                  <TableHead>Conteo</TableHead>
                  <TableHead>Porcentaje</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {Object.entries(resultsData).map(([category, count]) => {
                  const percentage =
                    count > 0
                      ? ((count / calculatePercentage()) * 100).toFixed(2)
                      : 0;
                  return (
                    <TableRow key={category}>
                      <TableCell>{category}</TableCell>
                      <TableCell>
                        <Badge
                          variant={
                            count > 0
                              ? count > 5
                                ? "warning"
                                : "default"
                              : "destructive"
                          }
                        >
                          {count}
                        </Badge>
                      </TableCell>
                      <TableCell>{percentage}%</TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>
        </div>
      )}
    </div>
  );
};

export default HomeApp;
