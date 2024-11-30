import React, { useState } from 'react';
import { Formik, Form } from 'formik';
import * as Yup from 'yup';
import { Download } from 'lucide-react';

const FileUploadSchema = Yup.object().shape({
  uploadFile: Yup.mixed()
    .required('Por favor selecciona un archivo')
    .test('fileType', 'Solo se permiten archivos CSV y Excel', (value) => {
      return value && (
        value.name.toLowerCase().endsWith('.csv') || 
        value.name.toLowerCase().endsWith('.xlsx') || 
        value.name.toLowerCase().endsWith('.xls')
      );
    })
    .test('fileSize', 'El archivo no debe exceder 10 MB', (value) => {
      return value && value.size <= 10 * 1024 * 1024; // 10 MB
    })
});

const FileTable = () => {
  const [processedData, setProcessedData] = useState([]);

  const addFile = async (newFile) => {
    try {
      // Simular respuesta de API
      const mockApiResponse = [
        {
          stockDiaActual: 100,
          pedidoEnTransito: 'extend intuitive partnerships',
          pedidoProcesandose: '8.89E+12',
          productDescription: 'XBIGGB55G9M6386',
          productId: 'Lopez Prairie',
          storeId: 'Mortonville, WY 96411',
          storeDescription: 'Sullivan-Hughes',
          categoryId: 'CR##########',
          pais: '101200',
          fechaSemana: 'exploit value-added mindshare',
          etiqueta: 'Etiqueta 1',
          observacion: 'Observación 1'
        }
      ];

      // Simular tiempo de procesamiento
      await new Promise(resolve => setTimeout(resolve, 2000));

      setProcessedData(mockApiResponse);

    } catch (error) {
      console.error('Error al procesar el archivo:', error);
    }
  };

  return (
    <div className="space-y-6">
      <Formik
        initialValues={{ uploadFile: null }}
        validationSchema={FileUploadSchema}
        onSubmit={(values, { setSubmitting, resetForm }) => {
          if (values.uploadFile) {
            addFile(values.uploadFile);
            resetForm();
          }
          setSubmitting(false);
        }}
      >
        {({ errors, touched, setFieldValue, isSubmitting }) => (
          <Form className="p-8 max-w-xl mx-auto bg-white shadow-md rounded-lg">
            <h2 className="text-2xl font-bold mb-6 text-center">Cargar Archivo CSV/Excel</h2>
            
            <div className="mb-4">
              <label 
                htmlFor="uploadFile" 
                className="block text-gray-700 text-sm font-bold mb-2"
              >
                Seleccionar Archivo CSV/Excel
              </label>
              <input
                id="uploadFile"
                name="uploadFile"
                type="file"
                accept=".csv,.xlsx,.xls"
                onChange={(event) => {
                  setFieldValue('uploadFile', event.currentTarget.files[0]);
                }}
                className="w-full p-2 border rounded-md"
              />
              {errors.uploadFile && touched.uploadFile && (
                <div className="text-red-500 text-sm mt-1">
                  {errors.uploadFile}
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

      {processedData.length > 0 && (
        <div className="p-8">
          <h2 className="text-2xl font-bold mb-6">Datos Procesados</h2>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse bg-white shadow-md rounded-lg overflow-hidden">
              <thead className="bg-gray-100">
                <tr>
                  {Object.keys(processedData[0]).map((key) => (
                    <th key={key} className="p-3 text-left whitespace-nowrap">
                      {key.charAt(0).toUpperCase() + key.slice(1).replace(/([A-Z])/g, ' $1')}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {processedData.map((item, rowIndex) => (
                  <tr key={rowIndex} className="border-b hover:bg-gray-50">
                    {Object.values(item).map((value, cellIndex) => (
                      <td key={cellIndex} className="p-3 whitespace-nowrap">
                        {value}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default FileTable;