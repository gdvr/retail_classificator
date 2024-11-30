import React from 'react';
import { Formik, Form } from 'formik';
import * as Yup from 'yup';
import { toast } from 'react-toastify';

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

const FileUpload = () => {
  const handleFileSubmit = async (values, { setSubmitting, resetForm }) => {
    const formData = new FormData();
    formData.append('csvFile', values.csvFile);

    try {
      // Simular envío de archivo a la API
      const mockApiResponse = await new Promise((resolve) => {
        setTimeout(() => {
          // Simular una clave de procesamiento única
          const processingKey = `process_${Date.now()}`;
          resolve({
            processingKey: processingKey,
            message: 'Archivo en procesamiento'
          });
        }, 1500);
      });

      // Guardar la clave de procesamiento en localStorage
      localStorage.setItem('processingKey', mockApiResponse.processingKey);

      // Mostrar notificación de éxito
      toast.success(mockApiResponse.message, {
        position: "top-right",
        autoClose: 3000,
      });

      setSubmitting(false);
      resetForm();
    } catch (error) {
      toast.error('Error al cargar el archivo', {
        position: "top-right",
        autoClose: 3000,
      });
      setSubmitting(false);
    }
  };

  return (
    <div className="p-8 max-w-xl mx-auto bg-white shadow-md rounded-lg mt-10">
      <h2 className="text-2xl font-bold mb-6 text-center">Cargar Archivo CSV</h2>
      
      <Formik
        initialValues={{ csvFile: null }}
        validationSchema={FileUploadSchema}
        onSubmit={handleFileSubmit}
      >
        {({ errors, touched, setFieldValue, isSubmitting }) => (
          <Form>
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
    </div>
  );
};

export default FileUpload;