import React, { useState } from 'react';
import { Formik, Form, Field } from 'formik';
import * as Yup from 'yup';
import { Download } from 'lucide-react';

const FileUploadSchema = Yup.object().shape({
  csvFile: Yup.mixed()
    .required('Por favor selecciona un archivo')
    .test('fileType', 'Solo se permiten archivos CSV', (value) => {
      return value && value.name.toLowerCase().endsWith('.csv');
    })
    .test('fileSize', 'El archivo no debe exceder 10 MB', (value) => {
      return value && value.size <= 10 * 1024 * 1024; // 10 MB
    })
});

// Componente de Tabla de Archivos
const FileTable = () => {
  const [files, setFiles] = useState([]);

  // Mapeo de estados a colores
  const statusColors = {
    'Pendiente': 'bg-red-500',
    'Procesándose': 'bg-yellow-500',
    'Finalizado': 'bg-green-500'
  };

  const addFile = (newFile) => {
    const fileWithStatus = {
      ...newFile,
      status: 'Pendiente',
      date: new Date().toLocaleDateString()
    };

    setFiles(prevFiles => [...prevFiles, fileWithStatus]);

    // Simular cambio de estado
    setTimeout(() => {
      setFiles(prevFiles => 
        prevFiles.map(file => 
          file.name === newFile.name 
            ? {...file, status: 'Procesándose'} 
            : file
        )
      );

      // Simular finalización
      setTimeout(() => {
        setFiles(prevFiles => 
          prevFiles.map(file => 
            file.name === newFile.name 
              ? {...file, status: 'Finalizado'} 
              : file
          )
        );
      }, 2000);
    }, 2000);
  };

  const handleFileDownload = (file) => {
    // Lógica de descarga simulada
    alert(`Descargando archivo: ${file.name}`);
  };

  return (
    <div className="space-y-6">
      <Formik
        initialValues={{ csvFile: null }}
        validationSchema={FileUploadSchema}
        onSubmit={(values, { setSubmitting, resetForm }) => {
          if (values.csvFile) {
            addFile(values.csvFile);
            resetForm();
          }
          setSubmitting(false);
        }}
      >
        {({ errors, touched, setFieldValue, isSubmitting }) => (
          <Form className="p-8 max-w-xl mx-auto bg-white shadow-md rounded-lg">
            <h2 className="text-2xl font-bold mb-6 text-center">Cargar Archivo CSV</h2>
            
            <div className="mb-4">
              <label 
                htmlFor="csvFile" 
                className="block text-gray-700 text-sm font-bold mb-2"
              >
                Seleccionar Archivo CSV
              </label>
              <input
                id="csvFile"
                name="csvFile"
                type="file"
                accept=".csv"
                onChange={(event) => {
                  setFieldValue('csvFile', event.currentTarget.files[0]);
                }}
                className="w-full p-2 border rounded-md"
              />
              {errors.csvFile && touched.csvFile && (
                <div className="text-red-500 text-sm mt-1">
                  {errors.csvFile}
                </div>
              )}
            </div>

            <button
              type="submit"
              disabled={isSubmitting}
              className={`
                w-full p-2 rounded-md text-white 
                ${isSubmitting 
                  ? 'bg-gray-400 cursor-not-allowed' 
                  : 'bg-blue-500 hover:bg-blue-600'
                }
              `}
            >
              {isSubmitting ? 'Cargando...' : 'Cargar Archivo'}
            </button>
          </Form>
        )}
      </Formik>

      {files.length > 0 && (
        <div className="p-8">
          <h2 className="text-2xl font-bold mb-6">Archivos Cargados</h2>
          <table className="w-full border-collapse bg-white shadow-md rounded-lg overflow-hidden">
            <thead className="bg-gray-100">
              <tr>
                <th className="p-3 text-left">Nombre del Archivo</th>
                <th className="p-3 text-left">Fecha de Carga</th>
                <th className="p-3 text-left">Estado</th>
                <th className="p-3 text-left">Acciones</th>
              </tr>
            </thead>
            <tbody>
              {files.map((file, index) => (
                <tr key={index} className="border-b hover:bg-gray-50">
                  <td className="p-3">{file.name}</td>
                  <td className="p-3">{file.date}</td>
                  <td className="p-3">
                    <span 
                      className={`
                        inline-block px-2 py-1 
                        text-white rounded 
                        ${statusColors[file.status]}
                      `}
                    >
                      {file.status}
                    </span>
                  </td>
                  <td className="p-3">
                    <button 
                      onClick={() => handleFileDownload(file)}
                      className="
                        bg-blue-500 text-white 
                        px-3 py-1 rounded 
                        hover:bg-blue-600 
                        flex items-center
                      "
                    >
                      <Download className="mr-2" size={16} />
                      Descargar
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default FileTable;