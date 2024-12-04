import React from 'react';
import { Formik, Form } from 'formik';
import * as Yup from 'yup';
import { toast } from 'react-toastify';
import axios from 'axios';
import { CloudCog } from 'lucide-react';

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
    formData.append('file', values.uploadFile); // Clave 'file' debe coincidir con la API

    try {
      // Enviar archivo a la API
      const response = await axios.post('http://localhost:8000/upload-file', formData);

      if (response.status === 200) {
        // Guardar información de éxito en localStorage si es necesario

        const { message } = response.data;
        // localStorage.setItem('processingKey', processingKey);

        // Mostrar notificación de éxito
        toast.success(message || 'Archivo cargado exitosamente.', {
          position: "top-right",
          autoClose: 3000,
        });
      } else {
        toast.error('Error al cargar el archivo. Intenta nuevamente.', {
          position: "top-right",
          autoClose: 3000,
        });
      }

      setSubmitting(false);
      resetForm();
    } catch (error) {
      console.log(error.response)
      console.error('Error al cargar el archivo:', error);
      toast.error('Error al cargar el archivo. Verifica la conexión con la API.', {
        position: "top-right",
        autoClose: 3000,
      });
      setSubmitting(false);
    }
  };

  return (
    <div className="p-8 max-w-xl mx-auto bg-white shadow-md rounded-lg mt-10">
      <h2 className="text-2xl font-bold mb-6 text-center">Cargar Archivo CSV/Excel</h2>
      
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